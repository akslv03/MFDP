import json
import logging
import uuid
import pika
from database.config import get_settings
from database.database import engine
from models.ml_task import MLTask, TaskStatus
from segmentation_task import do_task
from sqlmodel import Session

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

settings = get_settings()

worker_id = str(uuid.uuid4())[:8]
logger.info(f"Worker ID: {worker_id}")

connection_params = pika.ConnectionParameters(
    host=settings.RABBITMQ_HOST,
    port=settings.RABBITMQ_PORT,
    virtual_host="/",
    credentials=pika.PlainCredentials(
        username=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASS,
    ),
    heartbeat=0,
    blocked_connection_timeout=None,
)

connection = pika.BlockingConnection(connection_params)
channel = connection.channel()
queue_name = settings.RABBITMQ_QUEUE_NAME
channel.queue_declare(queue=queue_name, durable=True)
channel.basic_qos(prefetch_count=1)


def callback(ch, method, properties, body):
    task_id = None
    try:
        message = json.loads(body)
        task_id = message.get("task_id")
        features = message.get("features", {})
        model = message.get("model")
        timestamp = message.get("timestamp")

        logger.info(
            "Worker_id: %s. Task №%s. Model: %s. Time: %s",
            worker_id,
            task_id,
            model,
            timestamp,
        )

        with Session(engine) as session:
            task = session.get(MLTask, task_id)
            if task:
                task.status = TaskStatus.IN_PROGRESS
                task.error_message = None
                session.add(task)
                session.commit()

        image_path = features.get("image_path")
        patient_age = features.get("patient_age")
        patient_gender = features.get("patient_gender")
        if patient_age is not None:
            try:
                patient_age = int(patient_age)
            except (TypeError, ValueError):
                patient_age = None

        prediction = do_task(
            image_path=image_path,
            patient_age=patient_age,
            patient_gender=patient_gender,
        )

        logger.info("Task №%s completed", task_id)

        with Session(engine) as session:
            task = session.get(MLTask, task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.display_image_path = prediction.get("display_image_path")
                task.result_mask_path = prediction.get("mask_path")
                task.overlay_image_path = prediction.get("overlay_image_path")
                task.similarity_cases = prediction.get("similarity_cases")
                gallery = prediction.get("slice_gallery")
                if gallery is not None:
                    task.slice_gallery = (
                        gallery if isinstance(gallery, str) else json.dumps(gallery, ensure_ascii=False)
                    )
                task.error_message = None
                session.add(task)
                session.commit()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.exception("Error while processing message: %s", e)

        if task_id is not None:
            with Session(engine) as session:
                task = session.get(MLTask, task_id)
                if task:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    session.add(task)
                    session.commit()

        try:
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as ack_error:
            logger.error("Failed to ack message after error: %s", ack_error)


channel.basic_consume(
    queue=queue_name,
    on_message_callback=callback,
    auto_ack=False,
)

logger.info("Waiting for MRI segmentation tasks. To exit, press Ctrl+C")
channel.start_consuming()
