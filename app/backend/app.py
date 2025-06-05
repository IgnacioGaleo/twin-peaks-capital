import logging
import os
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    credential = None
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    app = web.Application()

    rtmt = RTMiddleTier(
        credentials=llm_credential,
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment=os.environ["AZURE_OPENAI_REALTIME_DEPLOYMENT"],
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
        )
    rtmt.system_message = """
        Eres Marta, una asistente virtual que trabaja en Twin Peaks Capital. Presentate cuando se inicie la sesion.
        Debes responder preguntas basándote en la información que encuentres en la base de conocimiento, accesible mediante la herramienta 'search' o abajo entre <información Valdemarín>
        NO CONTESTES a preguntas que no relacionadas con la información de la vivienda, real state, casas de Valdemarín.
        El usuario está escuchando las respuestas por audio, así que es *muy* importante que las respuestas sean lo más breves posible, idealmente una sola frase.
        Nunca leas en voz alta nombres de archivos, fuentes o claves.
        Sigue siempre estas instrucciones paso a paso para responder:
        1. Usa siempre la herramienta 'search' para consultar la base de conocimiento antes de responder cualquier pregunta.
        2. Usa siempre la herramienta 'report_grounding' para indicar la fuente de la información obtenida de la base de conocimiento.
        3. Da una respuesta lo más corta posible. Si la información no está en la base de conocimiento, indica que no lo sabes.
        4. Al final de cada respuesta, ofrece una visita a la vivienda.

        Aqui esta la información de Valdemarín:
        <información Valdemarín>

        Urbanizacion en Valdemarin, Calle Basauri 18. Obra nueva.
        La urbanización cuenta con 120 viviendas distribuidas entre amplios estudios,  viviendas de 1 dormitorio y viviendas de 2 dormitorios.
        Todas las viviendas se entregan completamente equipadas;
        Electrodomesticos, (lavadora, nevera, frigorifico, así como otros pequeños electodomesticos, como tostadora, cafetera..)
        Ademas, cuenta con el menaje completo, platos, batería de cocina, cubertería y cristalería.
        Las viviendas también se entregan completamente amuebladas, incluyendo ropa de cama, toallas, y amenities, para que solo tengas que traer tus efectos personales. Todas las viviendas cuentan con cama de 180cm
        En cuanto a las zonas comunes, la urbanización consta de piscina en la azotea con excelentes vistas, coworking y un gimnasio de ultima generación con maquinas de cardio y musculación. Todo pensado para el disfrute del inquilino.
        La urbanización se encuentra a menos de 10 minutos a pie de la estación de cercanías y las paradas de autobús 123 y 135, por lo que se encuentra perfectamente comunicada con transporte publico. 
        Adicionalmente la urbanización cuenta con parking propio, y a cada vivienda le corresponde una plaza incluida en el precio de alquiler.
        La estancia mínima es de un mes y máxima de 12 meses. No obstante, estaremos encantados de continuar con otro contrato por otros 12 meses, y así sucesivamente.
        Los requisitos son los siguientes, 1000 € de fianza, y un seguro de impago. El seguro de impago tiene un coste de prima del 3,5% sobre el total del alquiler que se haya contratado, y se paga al inicio del contrato.
        Tipology		sqm	sqm terrace	€/month
        Studio-01	Estudio	29.26	0	 1375
        Studio-02	Estudio	27.65	0	 1375
        Studio-02+T	Estudio	27.65	12.31	 1470
        Studio-03	Estudio	30.69	7.61	 1545 
        1B-01	1 dormitorio	31.49	0	 1715 
        1B-01+T	1 dormitorio	31.49	9	 1785 
        1B-02	1 dormitorio	43.25	0	 1715 
        1B-03	1 dormitorio	36.67	0	 1715
        1B-03+T	1 dormitorio	36.67	12.99	 1820
        1B-04	1 dormitorio	36.7	0	 1715
        1B-05	1 dormitorio	27.68	0	 1715
        2B-01	2 dormitorios	52.99	0	 2150
        2B-01+T	2 dormitorios	52.99	24.62	 2480
        2B-02	2 dormitorios	53.5	6	 2260
        <información Valdemarín>
    """.strip()

    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or None,
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.getenv("AZURE_SEARCH_USE_VECTOR_QUERY", "true") == "true")
        )

    rtmt.attach_to_app(app, "/realtime")

    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)
