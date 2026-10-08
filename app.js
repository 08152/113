const chat = document.getElementById("chat");
const input = document.getElementById("input");
const send = document.getElementById("send");

function addMessage(text, type) {
    const message = document.createElement("div");

    message.className = `message ${type}`;
    message.textContent = text;

    chat.appendChild(message);
    chat.scrollTop = chat.scrollHeight;

    return message;
}

function sendMessage() {
    const text = input.value.trim();

    if (!text) {
        return;
    }

    addMessage(text, "user");

    input.value = "";
    input.focus();

    // Temporärer Platzhalter.
    // Hier kommt später unser eigenes KI-Modell hin.
    setTimeout(() => {
        addMessage(
            "Meine KI wird gerade noch gebaut. Das neuronale Netzwerk kommt als Nächstes!",
            "ai"
        );
    }, 300);
}

send.addEventListener("click", sendMessage);

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        sendMessage();
    }
});
