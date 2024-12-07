from flask import Flask, request, send_file, jsonify
from gradio_client import Client
from io import BytesIO
import os

app = Flask(__name__)

@app.route("/")
def read_root():
    return jsonify({"status": "Server is running coba /kivotos?text="})

def generate_image_with_kivotos(prompt: str) -> BytesIO:
    client = Client("Linaqruf/kivotos-xl-2.0")
    
    print(f"Generating image for prompt: {prompt}")

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

    print("Result from Kivotos:", result)

    image_path = result[0][0].get('image')

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as file:
            image_bytes = file.read()
        return BytesIO(image_bytes)
    else:
        return None

@app.route("/kivotos", methods=["GET"])
def kivotos_endpoint():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    image_data = generate_image_with_kivotos(text)
    if image_data:
        return send_file(image_data, mimetype='image/png', as_attachment=True, download_name="output.png")
    else:
        return jsonify({"error": "Failed to generate image"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
