async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const captionEl = document.getElementById("caption");
    const preview = document.getElementById("preview");

    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image");
        return;
    }

    preview.src = URL.createObjectURL(file);
    captionEl.innerText = "⏳ Generating caption...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Backend error");
        }

        const data = await response.json();
        captionEl.innerText = "📝 " + data.caption;

    } catch (error) {
        console.error(error);
        captionEl.innerText = "❌ Error generating caption";
    }
}