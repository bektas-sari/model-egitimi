document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("#text-form");
    const textarea = document.querySelector("textarea");
    const button = document.querySelector("button");
    
    if (textarea) {
        textarea.addEventListener("input", () => {
            button.disabled = textarea.value.trim() === "";
        });
    }

    form.addEventListener("submit", () => {
        button.innerHTML = "Analiz Ediliyor...";
        button.disabled = true;
    });
});
