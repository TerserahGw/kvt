from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from gradio_client import Client
from io import BytesIO
from PIL import Image

app = FastAPI()

def generate_image_with_kivotos(prompt: str) -> BytesIO:
    client = Client("Linaqruf/kivotos-xl-2.0")
    result = client.predict(
        prompt=prompt,
        negative_prompt="nsfw, (low quality, worst quality:1.2), 3d, watermark, signature, ugly, poorly drawn",
        seed=0,
        custom_width=1024,
        custom_height=1024,
        guidance_scale=7,
        num_inference_steps=28,
        sampler="Euler a",
        aspect_ratio_selector="896 x 1152",
        use_upscaler=False,
        upscaler_strength=0.55,
        upscale_by=1.5,
        add_quality_tags=True,
        api_name="/run"
    )
    if isinstance(result, str):
        with open(result, "rb") as file:
            image_bytes = file.read()
    elif isinstance(result, Image.Image):
        image_io = BytesIO()
        result.save(image_io, format="PNG")
        image_bytes = image_io.getvalue()
    else:
        raise ValueError("Unexpected result type from Gradio API")
    return BytesIO(image_bytes)

@app.get("/kivotos")
async def kivotos_endpoint(text: str = Query(...)):
    try:
        image_data = generate_image_with_kivotos(text)
        return StreamingResponse(image_data, media_type="image/png", headers={"Content-Disposition": "inline; filename=output.png"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
