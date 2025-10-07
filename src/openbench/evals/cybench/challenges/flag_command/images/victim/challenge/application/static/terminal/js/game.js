import { displayLineInTerminal } from "./main.js";
import { GAME_LOST, GAME_WON } from "./commands.js";

// GAME MECHANICS
// ---------------------------------------
const timeDelay = 1000;

function displayGameResult(message, style) {
    setTimeout(() => {
        displayLineInTerminal({
            text: message,
            style: `${style} margin-right`,
            useTypingEffect: true,
            addPadding: true,
        });
    }, timeDelay);
}

export function playerLost() {
    displayGameResult(GAME_LOST, "error");
}

export function playerWon() {
    displayGameResult(GAME_WON, "success");
}