// Inicio del servidor --------------
var room = HBInit({
    roomName: "3v3 a los pases",
    maxPlayers: 12,
    public: true,
    noPlayer: true
});

var redTeamColors = { angle: 120, textColor: 0x000000, colors: [0xFF5C33 , 0xCF4B29 , 0xAD3F22] };
var blueTeamColors = { angle: 60, textColor: 0xFFFFFF, colors: [0x0C5363 , 0x07313B , 0x041C21] };

room.onRoomLink = function() {
    room.setTeamColors(1, redTeamColors.angle, redTeamColors.textColor, redTeamColors.colors);
    room.setTeamColors(2, blueTeamColors.angle, blueTeamColors.textColor, blueTeamColors.colors);
};

// URL y Mapas
const mapUrl = 'https://manoloBengo.github.io/haxmaps/x3_bazinga.hbs';
let originalMapData = null;
let isRevertingMap = false;
let isMatchActive = false;
let isBallBeingPlayed = false;
let mapChanged = false;
let afterGoal = false;

// --- CONFIGURACIÓN AFK ---
const AFK_CONFIG = {
    umbralInactividad: 600, // 10 segundos
    distanciaActivacion: 400, // Distancia para activar peligro (Normal)
    tiempoCuentaRegresiva: 480, // 8 segundos para reaccionar antes del kick
    tiempoAviso: 300, // Avisar cuando falten 5 segundos
};

let afkTracker = {}; 

function getDistance(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

// Carga de Mapa
function loadMapFromUrl(url, callback) {
    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error("Error al obtener el mapa");
            return response.text();
        })
        .then(data => callback(data))
        .catch(err => console.error("Error al cargar el mapa:", err));
}

loadMapFromUrl(mapUrl, (mapData) => {
    try {
        originalMapData = mapData;
        room.setCustomStadium(mapData);
        room.setScoreLimit(3);
        room.setTimeLimit(3);
        console.log("Mapa cargado y configurado correctamente.");
    } catch (error) {
        console.error("Error al configurar el mapa:", error);
    }
});

const hostPlayerId = 0;

// Logging y Backend
function logChatEvent(jugador, mensaje, tipo) {
    const evento = {
        hora: new Date().toISOString(),
        jugador: jugador || "SYSTEM",
        mensaje: mensaje,
        tipo: tipo
    };
    
    console.log(`${jugador} dice: ${mensaje}`)
    
    fetch("http://localhost:3000/guardar-chats", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(evento)
    }).catch(err => console.error("Error enviando evento de chat:", err));
}

// -----------------------------------------------------
// LÓGICA DE SUPLENTES (NUEVO)
// -----------------------------------------------------
function checkAndReplacePlayer(teamIdLeft) {
    // Si el que se fue era espectador (team 0), no hacemos nada
    if (teamIdLeft === 0) return;

    // Buscar espectadores disponibles
    const players = room.getPlayerList();
    const specs = players.filter(p => p.team === 0);

    if (specs.length > 0) {
        // --- CASO A: HAY SUPLENTE ---
        const substitute = specs[0]; // El primero de la lista
        
        // Mover al equipo
        room.setPlayerTeam(substitute.id, teamIdLeft);
        room.sendAnnouncement(`🔄 CAMBIO: ${substitute.name} entra a jugar. ¡Muévete o serás kickeado!`, null, 0x00FF00, "bold", 2);

        // Configurar AFK Estricto para el nuevo
        if (afkTracker[substitute.id]) {
            afkTracker[substitute.id].lastActivityTick = dataTickCounter;
            afkTracker[substitute.id].strictMode = true; 
        }

    } else {
        // --- CASO B: NO HAY NADIE (EQUIPO INCOMPLETO) ---
        // Pausamos el juego para mantener la integridad del 3v3
        room.pauseGame(true);
        room.sendAnnouncement("⚠️ Equipo incompleto y sin suplentes. Partido pausado.", null, 0xFF0000, "bold", 2);
    }
}
// -----------------------------------------------------

room.onPlayerJoin = function (player) {
    logChatEvent("SYSTEM", `Jugador ${player.name} entro`, "join");
    
    // Fetch al backend (tu código original)
    fetch("http://localhost:3000/get-or-create-unique-player", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_name: player.name })
    })
    .then(response => response.json())
    .then(data => {
        if (data.player_id) {
            player.customPlayerId = data.player_id;
        }
    })
    .catch(err => console.error("Error:", err));
    
    // Inicializar tracker AFK
    afkTracker[player.id] = {
        lastActivityTick: 0,
        isDanger: false,
        dangerCounter: 0,
        warned: false,
        strictMode: false // Por defecto es false
    };

    // Autocompletar equipos
    const players = room.getPlayerList();
    const redTeam = players.filter(p => p.team === 1);
    const blueTeam = players.filter(p => p.team === 2);

    // Si al Rojo le falta gente (menos de 3), lo metemos ahí
    if (redTeam.length < 3) {
        room.setPlayerTeam(player.id, 1);
        console.log(`${player.name} agregado automáticamente a Red.`);
    }
    // Si no, chequeamos el Azul
    else if (blueTeam.length < 3) {
        room.setPlayerTeam(player.id, 2);
        console.log(`${player.name} agregado automáticamente a Blue.`);
    }
};

room.onPlayerLeave = function (player) {
    logChatEvent("SYSTEM", `Jugador ${player.name} salio`, "leave");
    
    // GUARDAR EL EQUIPO ANTES DE BORRAR DATOS
    const teamLeft = player.team;

    // Verificar admins
    const players = room.getPlayerList();
    const admins = players.filter(p => p.admin && p.id !== hostPlayerId);
    if (admins.length === 0) console.log("No hay administradores.");

    delete afkTracker[player.id];

    // LLAMAR A LA FUNCIÓN DE SUPLENTES
    checkAndReplacePlayer(teamLeft);
};

room.onPlayerKicked = (player, reason, ban) => {
    let tipo = ban ? "ban" : "kick";
    logChatEvent("SYSTEM", `Jugador ${player.name} fue ${tipo}. Motivo: ${reason}`, tipo);
    // NOTA: onPlayerLeave se ejecuta inmediatamente después de esto, así que el reemplazo se maneja ahí.
};

room.onPlayerChat = function (player, message) {
    if (message === "!admin") {
        const players = room.getPlayerList();
        const admins = players.filter(p => p.admin && p.id !== hostPlayerId);
        
        if (admins.length === 0) {
            room.setPlayerAdmin(player.id, true);
            room.sendAnnouncement(`${player.name} ahora es administrador.`, null, 0xFFFF00, "bold", 2);
        } else {
            room.sendAnnouncement(`El comando '!admin' ya no esta disponible.`, player.id, 0xFF0000, "normal", 2);
        }
        return false;
    }
    if (player && player.name) logChatEvent(player.name, message, "mensaje");
    return true;
};

room.onStadiumChange = function (newStadiumName, byPlayer) {
    if (isMatchActive || isRevertingMap) return;
};

// --- FLUJO DE DATOS ---
var matchId = null;
let scorer = null;

function initializeMatchIdAndRegister() {
    fetch("http://localhost:3000/next-match-id")
        .then(res => res.json())
        .then(data => {
            matchId = data.next_match_id;
            registerMatchStart();
        })
        .catch(error => console.error("Error init match:", error));
}

function registerMatchStart() {
    const startTime = new Date().toISOString();
    fetch("http://localhost:3000/register-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ match_id: matchId, start_time: startTime }),
    }).then(() => isMatchActive = true);
}

function endMatch() {
    const endTime = new Date().toISOString();
    fetch("http://localhost:3000/register-end-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ match_id: matchId, end_time: endTime }),
    }).then(() => {
        isMatchActive = false;
        matchId = null;
    });
}

room.onGameStart = function () {
    if (!isMatchActive) initializeMatchIdAndRegister();
};

room.onGameStop = function () {
    if (isMatchActive) {
        isMatchActive = false;
        endMatch();
    }
    mapChanged = false;
};

// --- VARIABLES DE CONTROL Y LOOP ---
var playerPositions = [];
var sendInterval = 60;
const tickInterval = 60;

// IMPORTANTE: Variables separadas
let tickCounter = 0;      // Se reinicia cada 60 (para enviar datos)
let dataTickCounter = 0;  // NO se reinicia (para AFK)

let lastTouchPlayer = null;
let previousPositions = {};

function calculateVelocity(current, previous, deltaTime) {
    return {
        velocity_x: (current.x - previous.x) / deltaTime,
        velocity_y: (current.y - previous.y) / deltaTime,
    };
}

room.onGameTick = function () {
    if (!matchId) return;

    const players = room.getPlayerList();
    const ball = room.getBallPosition();
    const scores = room.getScores();
    const currentTime = scores ? scores.time : 0;

    if (ball.x === 0 && ball.y === 0) afterGoal = false;
    if (afterGoal && !isBallBeingPlayed) return;
    if (ball.x === 0 && ball.y === 0 && !isBallBeingPlayed) return;

    isBallBeingPlayed = true;

    if (!players || !ball) return;

    // Incrementamos ambos contadores
    tickCounter++;
    dataTickCounter++; 

    // --- LÓGICA AFK ---
    if (isMatchActive && ball) {
        players.forEach(p => {
            if (p.team === 0 || !afkTracker[p.id]) return;

            let tracker = afkTracker[p.id];

            // 1. Calcular inactividad usando el reloj GLOBAL (dataTickCounter)
            let ticksInactivos = dataTickCounter - tracker.lastActivityTick;
            
            if (ticksInactivos > AFK_CONFIG.umbralInactividad) {
                
                let dist = getDistance(p.position, ball);
                
                // --- CAMBIO AQUÍ PARA MODO ESTRICTO ---
                // Se activa si la pelota está cerca O si tiene strictMode activado (recién entró)
                if (dist < AFK_CONFIG.distanciaActivacion || tracker.strictMode) {
                    if (!tracker.isDanger) {
                        tracker.isDanger = true;
                        tracker.dangerCounter = 0; 
                    }
                }
            }

            // 2. Proceso de Cuenta Regresiva
            if (tracker.isDanger) {
                tracker.dangerCounter++;

                let ticksParaAviso = (AFK_CONFIG.tiempoCuentaRegresiva - (5 * 60));
                if (tracker.dangerCounter === ticksParaAviso && !tracker.warned) {
                    room.sendAnnouncement("⚠️ ¡Movete pibe! O en 5 segundos te rejamos.", p.id, 0xFFFF00, "bold", 2);
                    tracker.warned = true;
                }

                if (tracker.dangerCounter >= AFK_CONFIG.tiempoCuentaRegresiva) {
                    room.kickPlayer(p.id, "Por inactividad (AFK)", false);
                }
            }
            
            // Debug Log (Opcional, coméntalo si hay lag)
            // let dist = getDistance(p.position, ball);
            // console.log(`Tick: ${dataTickCounter} | ${p.name} | Strict: ${tracker.strictMode} | Danger: ${tracker.isDanger}`);
        });
    }

    // --- GUARDAR POSICIONES ---
    const validPlayers = players.filter(p => p.position).map(player => {
        const currentPos = { x: player.position.x, y: player.position.y, time: currentTime };
        const prev = previousPositions[player.id] || currentPos;
        const deltaTime = prev.time !== undefined ? currentTime - prev.time : 1;
        const velocity = calculateVelocity(currentPos, prev, deltaTime);
        previousPositions[player.id] = currentPos;

        return {
            player_id: player.id,
            player_name: player.name,
            x: currentPos.x,
            y: currentPos.y,
            velocity_x: velocity.velocity_x,
            velocity_y: velocity.velocity_y,
            team: player.team
        };
    });

    const ballData = (() => {
        const currentBall = { x: ball.x, y: ball.y, time: currentTime };
        const prevBall = previousPositions[0] || currentBall;
        const deltaTime = prevBall.time !== undefined ? currentTime - prevBall.time : 1;
        const velocity = calculateVelocity(currentBall, prevBall, deltaTime);
        previousPositions[0] = currentBall;

        return {
            player_id: 0,
            player_name: "Ball",
            x: currentBall.x,
            y: currentBall.y,
            velocity_x: velocity.velocity_x,
            velocity_y: velocity.velocity_y
        };
    })();

    const filteredPlayers = validPlayers.filter(p => p.x != null && p.y != null);
    
    if (ballData.x != null) {
        playerPositions.push({ time: currentTime, players: filteredPlayers, ball: ballData });
    }

    // Enviar datos al backend (Usa tickCounter para reiniciar ciclo)
    if (tickCounter >= tickInterval) {
        tickCounter = 0; // Reinicia SOLO el de envío
        sendDataToBackend(playerPositions);
        playerPositions = [];
    }

    // Detector de toque
    let closestPlayer = null;
    let closestDistance = Infinity;
    players.forEach(player => {
        if (player.position) {
            let distance = Math.sqrt(Math.pow(player.position.x - ball.x, 2) + Math.pow(player.position.y - ball.y, 2));
            if (distance < 50 && distance < closestDistance) {
                closestDistance = distance;
                closestPlayer = player;
            }
        }
    });
    if (closestPlayer) lastTouchPlayer = closestPlayer;
};

// Función para enviar datos
function sendDataToBackend(data) {
    if (!matchId) return;
    fetch("http://localhost:3000/save-positions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ match_id: matchId, positions: data , hora: new Date().toISOString()}),
    }).catch(err => console.error("Error enviando posiciones:", err));
}

room.onPlayerBallTouch = function (player) { lastTouchPlayer = player; };

room.onTeamGoal = function (team) {
    let scoringTeam = (team === 1) ? "Red" : "Blue";
    isBallBeingPlayed = false;
    afterGoal = true;
    const scores = room.getScores();
    const goalTick = scores ? scores.time : 0;
    const hora_gol = new Date().toISOString();

    if (lastTouchPlayer) {
        console.log(`GOOOOL de ${lastTouchPlayer.name}`);
        fetch("http://localhost:3000/save-goal", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                match_id: matchId, player_id: lastTouchPlayer.id, player_name: lastTouchPlayer.name,
                equipo: scoringTeam, tick: goalTick, hora: hora_gol
            }),
        }).catch(err => console.error("Error enviando gol:", err));
    }
};

room.onTeamVictory = function (scores) {
    isBallBeingPlayed = false;
    console.log(`Victoria: ${scores.red > scores.blue ? 'Red' : 'Blue'}`);
};

// --- ACTIVIDAD DEL JUGADOR ---
room.onPlayerActivity = function(player) {
    if (afkTracker[player.id]) {
        
        // CORRECCIÓN IMPORTANTE: Usar dataTickCounter (el que no se reinicia)
        afkTracker[player.id].lastActivityTick = dataTickCounter; 
        
        // Si tenía modo estricto (acababa de entrar), se lo quitamos
        // porque ya demostró que está vivo
        if (afkTracker[player.id].strictMode) {
            afkTracker[player.id].strictMode = false;
            // console.log(`${player.name} ya no está en modo estricto.`);
        }

        // Si estaba en peligro, se salva
        if (afkTracker[player.id].isDanger) {
            afkTracker[player.id].isDanger = false;
            afkTracker[player.id].dangerCounter = 0;
            afkTracker[player.id].warned = false;
            console.log(`${player.name} se salvó del AFK!`);
        }
    }
};