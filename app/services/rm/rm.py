import pika
import json
from database.config import get_settings

settings = get_settings()

connection_params = pika.ConnectionParameters(
    host=settings.RABBITMQ_HOST,
    port=settings.RABBITMQ_PORT,
    virtual_host='/',
    credentials=pika.PlainCredentials(
        username=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASS
    ),
    heartbeat=30,
    blocked_connection_timeout=2
)

def send_task(message: dict):
    """Кладет задачу сегментации в очередь RabbitMQ."""
    connection = pika.BlockingConnection(connection_params)
    try:
        channel = connection.channel()

        queue_name = settings.RABBITMQ_QUEUE_NAME
        channel.queue_declare(queue=queue_name, durable=True)

        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2),
        )
    finally:
        connection.close()
