import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="Tankri Dataset Builder V2")

st.title("Tankri Dataset Builder V2")

os.makedirs("dataset/images", exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload Page",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.subheader("Step 1: Draw a rectangle around the Tankri table")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="canvas",
    )

    rows = st.number_input(
        "Rows",
        min_value=1,
        value=10
    )

    cols = st.number_input(
        "Columns",
        min_value=1,
        value=5
    )

    if st.button("Extract Table"):

        if (
            canvas_result.json_data
            and len(canvas_result.json_data["objects"]) > 0
        ):

            rect = canvas_result.json_data["objects"][0]

            x = int(rect["left"])
            y = int(rect["top"])
            w = int(rect["width"] * rect["scaleX"])
            h = int(rect["height"] * rect["scaleY"])

            table = image.crop(
                (
                    x,
                    y,
                    x + w,
                    y + h
                )
            )

            cell_width = w // cols
            cell_height = h // rows

            counter = 1

            for r in range(rows):
                for c in range(cols):

                    left = c * cell_width
                    top = r * cell_height

                    right = left + cell_width
                    bottom = top + cell_height

                    crop = table.crop(
                        (
                            left,
                            top,
                            right,
                            bottom
                        )
                    )

                    crop.save(
                        f"dataset/images/{counter}.png"
                    )

                    counter += 1

            st.success(
                f"Saved {counter - 1} images!"
            )

        else:
            st.error("Please draw a rectangle first.")