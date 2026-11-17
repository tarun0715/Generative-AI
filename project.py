import streamlit as st
from google import genai
from PIL import Image, UnidentifiedImageError
import os


# ----------------------------------
# 1. STREAMLIT CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Gemini Image Analyzer",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Gemini Image Analyzer")
st.write("Upload an image and ask Gemini to analyze it.")


# ----------------------------------
# 2. GEMINI API CONFIGURATION
# ----------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. "
        "Please configure GEMINI_API_KEY."
    )
    st.stop()

try:
    client = genai.Client(api_key=api_key)

except Exception as e:
    st.error(f"Gemini initialization failed: {e}")
    st.stop()


# ----------------------------------
# 3. GEMINI RESPONSE FUNCTION
# ----------------------------------

def get_gemini_response(prompt, image):

    model = "gemini-3.8-flash"

    contents = []

    if prompt.strip():
        contents.append(prompt)
    else:
        contents.append("Describe this image in detail.")

    contents.append(image)

    response = client.models.generate_content(
        model=model,
        contents=contents
    )

    if response.text:
        return response.text

    return "Gemini did not return any text."


# ----------------------------------
# 4. USER INPUT
# ----------------------------------

prompt = st.text_input(
    "Enter your prompt",
    placeholder="Describe this image..."
)


# ----------------------------------
# 5. IMAGE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image.verify()

        # Reopen after verification
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    except (UnidentifiedImageError, OSError):
        st.error("Invalid or corrupted image file.")

    except Exception as e:
        st.error(f"Image processing error: {e}")


# ----------------------------------
# 6. SUBMIT BUTTON
# ----------------------------------

if st.button("Analyze Image", type="primary"):

    if uploaded_file is None or image is None:

        st.warning("Please upload a valid image first.")

    else:

        try:
            with st.spinner("Gemini is analyzing your image..."):

                response = get_gemini_response(
                    prompt,
                    image
                )

            st.subheader("Gemini Response")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")


# ----------------------------------
# 7. FOOTER
# ----------------------------------

st.divider()
st.caption("Powered by Google Gemini and Streamlit")
