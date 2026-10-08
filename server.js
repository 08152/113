const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 10000;
const HOST = "0.0.0.0";

const server = http.createServer((req, res) => {
    let requestPath = req.url.split("?")[0];

    if (requestPath === "/") {
        requestPath = "/index.html";
    }

    // Sicherheitsprüfung gegen einfache Path-Traversal-Versuche
    const safePath = path.normalize(requestPath).replace(/^(\.\.[\/\\])+/, "");

    const filePath = path.join(
        __dirname,
        safePath
    );

    const allowedFiles = [
        path.join(__dirname, "index.html"),
        path.join(__dirname, "app.js")
    ];

    if (!allowedFiles.includes(filePath)) {
        res.writeHead(404, {
            "Content-Type": "text/plain; charset=utf-8"
        });

        res.end("404 - Datei nicht gefunden");
        return;
    }

    fs.readFile(filePath, (error, data) => {

        if (error) {
            res.writeHead(500, {
                "Content-Type": "text/plain; charset=utf-8"
            });

            res.end("500 - Serverfehler");
            return;
        }

        let contentType = "text/plain; charset=utf-8";

        if (filePath.endsWith(".html")) {
            contentType = "text/html; charset=utf-8";
        }

        if (filePath.endsWith(".js")) {
            contentType = "text/javascript; charset=utf-8";
        }

        res.writeHead(200, {
            "Content-Type": contentType
        });

        res.end(data);
    });
});

server.listen(PORT, HOST, () => {
    console.log(`KI-Server läuft auf http://${HOST}:${PORT}`);
});
