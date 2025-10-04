<?php

// read POST body as JSON
$_DATA = json_decode(file_get_contents('php://input'), true);

// set output as JSON
header('Content-Type: application/json');

function json_die($message, $data, $code = 500) {
    http_response_code($code);
    die(json_encode([
        "error" => [
            "code" => $code,
            "message" => $message,
            "data" => $data
        ]
    ]));
}

function json_response($message, $data = null) {
    die(json_encode([
        "message" => $message,
        "data" => $data
    ]));
}
