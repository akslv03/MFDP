import json
import logging
import time
import uuid
import pika
from pika.exceptions import AMQPConnectionError, StreamLostError
from database.config import get_settings
from database.database import engine
from models.ml_task import MLTask, TaskStatus
from segmentation_task import do_task
from sqlmodel import Session
from sqlalchemy.exc import OperationalError

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

settings = get_settings()

worker_id = str(uuid.uuid4())[:8]
logger.info("Worker ID: %s", worker_id)

RECONNECT_DELAY_SEC = 5


def _connection_params() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host="/",
        credentials=pika.PlainCredentials(
            username=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASS,
        ),
        heartbeat=60,
        blocked_connection_timeout=300,
        connection_attempts=3,
        retry_delay=2,
    )


def _wait_for_database(retries: int = 30, delay_sec: float = 2.0) -> None:
    """Ждём готовности Postgres перед стартом consumer."""
    for attempt in range(1, retries + 1):
        try:
            with Session(engine) as session:
                session.connection()
            logger.info("PostgreSQL is ready")
            return
        except OperationalError as exc:
            logger.warning(
                "PostgreSQL not ready (attempt %s/%s): %s",
                attempt,
                retries,
                exc,
            )
            time.sleep(delay_sec)
    raise RuntimeError("PostgreSQL did not become ready in time")


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
            try:
                with Session(engine) as session:
                    task = session.get(MLTask, task_id)
                    if task:
                        task.status = TaskStatus.FAILED
                        task.error_message = str(e)
                        session.add(task)
                        session.commit()
            except Exception as db_error:
                logger.error("Failed to mark task as FAILED: %s", db_error)

        try:
            if ch.is_open:
                ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as ack_error:
            logger.error("Failed to ack message after error: %s", ack_error)


def run_consumer() -> None:
    _wait_for_database()
    queue_name = settings.RABBITMQ_QUEUE_NAME

    while True:
        connection = None
        try:
            connection = pika.BlockingConnection(_connection_params())
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=False,
            )
            logger.info("Waiting for MRI segmentation tasks. To exit, press Ctrl+C")
            channel.start_consuming()
        except (AMQPConnectionError, StreamLostError, OSError) as exc:
            logger.error("RabbitMQ connection lost: %s. Reconnect in %ss", exc, RECONNECT_DELAY_SEC)
            time.sleep(RECONNECT_DELAY_SEC)
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        finally:
            if connection is not None and connection.is_open:
                try:
                    connection.close()
                except Exception:
                    pass


if __name__ == "__main__":
    run_consumer()
