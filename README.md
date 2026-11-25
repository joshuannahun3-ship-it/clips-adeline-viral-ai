# Clips Adeline Viral AI

Aplicación inicial para crear clips desde videos de YouTube que tengas derecho a reutilizar.

## Funciones de esta primera versión
- Pegar un enlace de YouTube.
- Descargar videos públicos compatibles con yt-dlp.
- Transcribir audio con OpenAI Whisper.
- Proponer segmentos hablados con una puntuación heurística inicial.
- Exportar MP4 vertical 9:16 y horizontal 16:9.
- Mantener el audio original y añadir subtítulos.

## Despliegue en Render
Este repositorio incluye `render.yaml` y `Dockerfile`.
1. En Render, crea un servicio nuevo desde el repositorio `clips-adeline-viral-ai`.
2. Render detectará el Blueprint `render.yaml`.
3. Configura `OPENAI_API_KEY` en las variables de entorno del servicio. No la publiques en GitHub.
4. Espera el despliegue y prueba `/health`.

## Limitaciones importantes
- Requiere una clave API de OpenAI con facturación habilitada para transcripción.
- La puntuación es heurística, no una predicción real de viralidad.
- No detecta automáticamente escenas visuales ni genera seguimiento inteligente de rostros en esta versión.
- Los directos deben haber terminado y ser accesibles.
- El acceso/descarga de videos depende de YouTube, yt-dlp, derechos de uso y restricciones del contenido.
- Esta versión procesa una solicitud de forma síncrona; no es todavía una plataforma de producción escalable.
