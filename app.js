// ========================================
// Eigene kleine KI
// Einfaches neuronales Netzwerk
// ========================================

const chat = document.getElementById("chat");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");

const trainQuestion = document.getElementById("trainQuestion");
const trainAnswer = document.getElementById("trainAnswer");
const trainButton = document.getElementById("trainButton");
const resetButton = document.getElementById("resetButton");
const status = document.getElementById("status");


// ========================================
// Trainingsdaten
// ========================================

const defaultData = [
    {
        text: "hallo",
        answer: "Hallo! Schön, dass du da bist."
    },
    {
        text: "hi",
        answer: "Hi! Wie geht es dir?"
    },
    {
        text: "hey",
        answer: "Hey! Was möchtest du wissen?"
    },

    {
        text: "wie geht es dir",
        answer: "Mir geht es gut. Ich bin bereit zum Lernen!"
    },
    {
        text: "wie gehts",
        answer: "Gut! Danke der Nachfrage."
    },

    {
        text: "wie heißt du",
        answer: "Ich bin deine eigene kleine KI."
    },
    {
        text: "wer bist du",
        answer: "Ich bin eine kleine KI, die mit JavaScript programmiert wurde."
    },

    {
        text: "was kannst du",
        answer: "Ich kann Texte erkennen und aus meinen Trainingsdaten passende Antworten auswählen."
    },

    {
        text: "tschüss",
        answer: "Tschüss! Bis später."
    },
    {
        text: "auf wiedersehen",
        answer: "Bis bald!"
    },

    {
        text: "danke",
        answer: "Gerne!"
    },
    {
        text: "dankeschön",
        answer: "Sehr gerne."
    },

    {
        text: "was ist ki",
        answer: "KI bedeutet künstliche Intelligenz. Sie kann bestimmte Aufgaben mithilfe von Algorithmen und trainierten Daten lösen."
    },

    {
        text: "was ist javascript",
        answer: "JavaScript ist eine Programmiersprache, die unter anderem Webseiten interaktiv macht."
    }
];


// ========================================
// Trainingsdaten laden
// ========================================

let trainingData;

const savedData = localStorage.getItem("my_ai_training");

if (savedData) {
    try {
        trainingData = JSON.parse(savedData);
    } catch {
        trainingData = [...defaultData];
    }
} else {
    trainingData = [...defaultData];
}


// ========================================
// Tokenizer
// ========================================

function tokenize(text) {
    return text
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, "")
        .split(/\s+/)
        .filter(Boolean);
}


// ========================================
// Vokabular erstellen
// ========================================

function buildVocabulary(data) {

    const words = new Set();

    for (const item of data) {
        const tokens = tokenize(item.text);

        for (const word of tokens) {
            words.add(word);
        }
    }

    return [...words];
}


// ========================================
// Ein Satz -> Zahlen
// ========================================

function vectorize(text, vocabulary) {

    const tokens = tokenize(text);

    return vocabulary.map(word => {
        return tokens.includes(word) ? 1 : 0;
    });
}


// ========================================
// Zufallszahl
// ========================================

function randomWeight() {
    return (Math.random() - 0.5) * 2;
}


// ========================================
// Mathematische Funktionen
// ========================================

function tanh(x) {
    return Math.tanh(x);
}

function tanhDerivative(x) {
    const t = Math.tanh(x);
    return 1 - t * t;
}

function softmax(values) {

    const max = Math.max(...values);

    const expValues = values.map(x =>
        Math.exp(x - max)
    );

    const sum = expValues.reduce(
        (a, b) => a + b,
        0
    );

    return expValues.map(x => x / sum);
}


// ========================================
// KI-Klasse
// ========================================

class NeuralNetwork {

    constructor(inputSize, hiddenSize, outputSize) {

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Input -> Hidden
        this.weights1 = Array.from(
            { length: inputSize },
            () => Array.from(
                { length: hiddenSize },
                randomWeight
            )
        );

        this.bias1 = Array.from(
            { length: hiddenSize },
            randomWeight
        );

        // Hidden -> Output
        this.weights2 = Array.from(
            { length: hiddenSize },
            () => Array.from(
                { length: outputSize },
                randomWeight
            )
        );

        this.bias2 = Array.from(
            { length: outputSize },
            randomWeight
        );
    }


    forward(input) {

        // Hidden-Schicht

        const hiddenRaw =
            Array(this.hiddenSize).fill(0);

        for (let h = 0; h < this.hiddenSize; h++) {

            let sum = this.bias1[h];

            for (let i = 0; i < this.inputSize; i++) {
                sum += input[i] * this.weights1[i][h];
            }

            hiddenRaw[h] = sum;
        }

        const hidden =
            hiddenRaw.map(tanh);


        // Output-Schicht

        const outputRaw =
            Array(this.outputSize).fill(0);

        for (let o = 0; o < this.outputSize; o++) {

            let sum = this.bias2[o];

            for (let h = 0; h < this.hiddenSize; h++) {
                sum += hidden[h] * this.weights2[h][o];
            }

            outputRaw[o] = sum;
        }

        const output =
            softmax(outputRaw);

        return {
            hiddenRaw,
            hidden,
            outputRaw,
            output
        };
    }


    train(input, target, learningRate = 0.08) {

        const result = this.forward(input);

        const hidden = result.hidden;
        const hiddenRaw = result.hiddenRaw;
        const output = result.output;


        // Output-Fehler

        const outputError =
            output.map(
                (value, index) =>
                    value - target[index]
            );


        // Hidden-Fehler

        const hiddenError =
            Array(this.hiddenSize).fill(0);

        for (let h = 0; h < this.hiddenSize; h++) {

            let error = 0;

            for (let o = 0; o < this.outputSize; o++) {

                error +=
                    outputError[o] *
                    this.weights2[h][o];
            }

            hiddenError[h] =
                error * tanhDerivative(hiddenRaw[h]);
        }


        // Gewichte Hidden -> Output

        for (let h = 0; h < this.hiddenSize; h++) {

            for (let o = 0; o < this.outputSize; o++) {

                this.weights2[h][o] -=
                    learningRate *
                    hidden[h] *
                    outputError[o];
            }
        }


        // Output Bias

        for (let o = 0; o < this.outputSize; o++) {

            this.bias2[o] -=
                learningRate *
                outputError[o];
        }


        // Gewichte Input -> Hidden

        for (let i = 0; i < this.inputSize; i++) {

            for (let h = 0; h < this.hiddenSize; h++) {

                this.weights1[i][h] -=
                    learningRate *
                    input[i] *
                    hiddenError[h];
            }
        }


        // Hidden Bias

        for (let h = 0; h < this.hiddenSize; h++) {

            this.bias1[h] -=
                learningRate *
                hiddenError[h];
        }
    }
}


// ========================================
// KI neu erstellen
// ========================================

let vocabulary = [];
let answers = [];
let network;


function buildAI() {

    vocabulary = buildVocabulary(trainingData);

    answers = [
        ...new Set(
            trainingData.map(item => item.answer)
        )
    ];

    const hiddenSize = Math.max(
        12,
        Math.min(40, vocabulary.length)
    );

    network = new NeuralNetwork(
        vocabulary.length,
        hiddenSize,
        answers.length
    );


    // Trainingsschleifen

    const epochs = 1200;

    for (let epoch = 0; epoch < epochs; epoch++) {

        for (const item of trainingData) {

            const input =
                vectorize(item.text, vocabulary);

            const target =
                Array(answers.length).fill(0);

            const answerIndex =
                answers.indexOf(item.answer);

            target[answerIndex] = 1;

            network.train(
                input,
                target
            );
        }
    }

    status.textContent =
        `KI trainiert: ${trainingData.length} Beispiele, ${vocabulary.length} Wörter`;
}


// ========================================
// Antwort finden
// ========================================

function getAnswer(text) {

    if (!text.trim()) {
        return "Schreib bitte etwas.";
    }

    const input =
        vectorize(text, vocabulary);

    // Kein bekanntes Wort
    const hasKnownWord =
        input.some(value => value === 1);

    if (!hasKnownWord) {
        return "Das kenne ich noch nicht. Du kannst mich unten trainieren.";
    }


    const result =
        network.forward(input);

    const probabilities =
        result.output;


    let bestIndex = 0;

    for (let i = 1; i < probabilities.length; i++) {

        if (
            probabilities[i] >
            probabilities[bestIndex]
        ) {
            bestIndex = i;
        }
    }


    const confidence =
        probabilities[bestIndex];


    // Unsicherheit erkennen

    if (confidence < 0.50) {
        return "Ich bin mir bei dieser Frage noch nicht sicher.";
    }

    return answers[bestIndex];
}


// ========================================
// Nachricht anzeigen
// ========================================

function addMessage(text, type) {

    const div =
        document.createElement("div");

    div.className =
        `message ${type}`;

    div.textContent = text;

    chat.appendChild(div);

    chat.scrollTop =
        chat.scrollHeight;
}


// ========================================
// Nachricht senden
// ========================================

function sendMessage() {

    const text =
        messageInput.value.trim();

    if (!text) {
        return;
    }

    addMessage(text, "user");

    const answer =
        getAnswer(text);

    setTimeout(() => {
        addMessage(answer, "ai");
    }, 200);

    messageInput.value = "";

    messageInput.focus();
}


sendButton.addEventListener(
    "click",
    sendMessage
);


messageInput.addEventListener(
    "keydown",
    event => {

        if (event.key === "Enter") {
            sendMessage();
        }
    }
);


// ========================================
// KI trainieren
// ========================================

trainButton.addEventListener(
    "click",
    () => {

        const question =
            trainQuestion.value.trim();

        const answer =
            trainAnswer.value.trim();


        if (!question || !answer) {

            alert(
                "Bitte Frage und Antwort eingeben."
            );

            return;
        }


        trainingData.push({
            text: question,
            answer: answer
        });


        localStorage.setItem(
            "my_ai_training",
            JSON.stringify(trainingData)
        );


        trainQuestion.value = "";
        trainAnswer.value = "";


        status.textContent =
            "Training läuft...";


        // Kurz warten, damit die Anzeige aktualisiert wird

        setTimeout(() => {

            buildAI();

            addMessage(
                "Ich habe etwas Neues gelernt! 🤖",
                "ai"
            );

        }, 50);
    }
);


// ========================================
// Zurücksetzen
// ========================================

resetButton.addEventListener(
    "click",
    () => {

        const confirmed =
            confirm(
                "Gesamtes eigenes Training löschen?"
            );

        if (!confirmed) {
            return;
        }


        trainingData =
            [...defaultData];


        localStorage.removeItem(
            "my_ai_training"
        );


        buildAI();


        addMessage(
            "Mein eigenes Training wurde zurückgesetzt.",
            "ai"
        );
    }
);


// ========================================
// Start
// ========================================

buildAI();
