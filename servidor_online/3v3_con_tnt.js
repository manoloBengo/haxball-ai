// Inicio del servidor --------------
var room = HBInit({
    roomName: "3v3 picante ??",
    maxPlayers: 9,
    public: true,
    noPlayer: true
});

var redTeamColors = { angle: 120, textColor: 0x000000, colors: [0xFF5C33 , 0xCF4B29 , 0xAD3F22] }; // Camiseta roja con degradado
var blueTeamColors = { angle: 60, textColor: 0xFFFFFF, colors: [0x0C5363 , 0x07313B , 0x041C21] }; // Camiseta azul con degradado

// Aplicar los colores cuando el servidor se abre
room.onRoomLink = function() {
    room.setTeamColors(1, redTeamColors.angle, redTeamColors.textColor, redTeamColors.colors);
    room.setTeamColors(2, blueTeamColors.angle, blueTeamColors.textColor, blueTeamColors.colors);
};


// URL de descarga directa del archivo del mapa (almacenado en mi github)
const mapUrl = 'https://manoloBengo.github.io/haxmaps/x3_bazinga.hbs';
let originalMapData = null;
let isRevertingMap = false; // Bandera para evitar bucles infinitos de cambios de mapa
let isMatchActive = false; // Bandera para saber si un partido est芍 activo
let isBallBeingPlayed = false; // Bandera para saber si un partido est芍 activo
let mapChanged = false; // Bandera para evitar m迆ltiples cambios de mapa
let afterGoal = false; // Bandera para saber si el saque de medio es despues de un gol

// Funci車n para cargar el mapa desde una URL
function loadMapFromUrl(url, callback) {
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error("Error al obtener el mapa");
            }
            return response.text();
        })
        .then(data => {
            callback(data);
        })
        .catch(err => console.error("Error al cargar el mapa desde la URL:", err));
}

// Cargar el mapa y configurarlo al iniciar el servidor
loadMapFromUrl(mapUrl, (mapData) => {
    try {
        originalMapData = mapData; // Guardar el mapa original
        room.setCustomStadium(mapData);
        room.setScoreLimit(3);
        room.setTimeLimit(3);
        console.log("Mapa cargado y configurado correctamente.");
    } catch (error) {
        console.error("Error al configurar el mapa:", error);
    }
});

// Comandos --------------------------

const hostPlayerId = 0; // ID del host, que siempre est芍 y no debe ser manejado

// Mensaje cuando un jugador entra
room.onPlayerJoin = function (player) {
    fetch("http://localhost:3000/get-or-create-unique-player", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_name: player.name })
    })
        .then(response => response.json())
        .then(data => {
            if (data.player_id) {
                console.log(`Jugador ${player.name} asignado con ID 迆nico: ${data.player_id}`);
                player.customPlayerId = data.player_id; // Guarda el ID 迆nico para el jugador
            } else {
                console.error(`No se pudo asignar un ID al jugador ${player.name}`);
            }
        })
        .catch(err => console.error("Error al obtener o crear el jugador:", err));
};

// Mensaje cuando un jugador sale
room.onPlayerLeave = function (player) {
    console.log(`${player.name} salio de la sala.`);
    
    // Verificar si alg迆n administrador sigue en la sala
    const players = room.getPlayerList();
    const admins = players.filter(p => p.admin && p.id !== hostPlayerId);
    
    if (admins.length === 0) {
        console.log("No hay administradores en la sala. Se puede reclamar admin nuevamente.");
    }
};

// Capturar mensajes de chat
room.onPlayerChat = function (player, message) {
    // Comando para obtener admin
    if (message === "!admin") {
        const players = room.getPlayerList();
        const admins = players.filter(p => p.admin && p.id !== hostPlayerId);
        
        if (admins.length === 0) {
            room.setPlayerAdmin(player.id, true); // Asigna admin al jugador
            room.sendAnnouncement(`${player.name} ahora es administrador.`, null, 0xFFFF00, "bold", 2);
            console.log(`${player.name} ha sido asignado como administrador.`);
        } else {
            room.sendAnnouncement(`El comando '!admin' ya no esta disponible, alguien ya tiene admin.`, player.id, 0xFF0000, "normal", 2);
        }
        return false; // No mostrar el mensaje en el chat p迆blico
    }
    
    // Bloquear intentos de cambiar el mapa
    if ((message.startsWith("!map ") || message.startsWith("/map ")) && player.id !== hostPlayerId) {
        room.sendAnnouncement(`No tienes permiso para cambiar el mapa.`, player.id, 0xFF0000, "normal", 2);
        return false; // No mostrar el mensaje en el chat p迆blico
    }
    
    return true; // Permite que el mensaje se muestre en el chat
};

// Interceptar cambios de mapa
room.onStadiumChange = function (newStadiumName, byPlayer) {
    if (isMatchActive || isRevertingMap) return; // Evitar cambios durante el partido y bucles infinitos

    if (byPlayer && byPlayer.id !== hostPlayerId) {
        // Revertir al mapa original si el cambio no lo hizo el host
        if (!mapChanged) {
            isRevertingMap = true;
            setTimeout(() => {
                room.setCustomStadium(originalMapData);
                isRevertingMap = false;
                mapChanged = true;
                room.sendAnnouncement(`Intento de cambio de mapa por ${byPlayer.name} bloqueado.`, null, 0xFF0000, "normal", 2);
            }, 100); // A?adir un peque?o retraso para asegurar que el cambio se procese correctamente
            console.log(`Intento de cambio de mapa por ${byPlayer.name} bloqueado.`);
        }
    } else {
        mapChanged = false; // Resetear la bandera si el cambio lo hizo el host
        console.log(`Cambio de mapa realizado por el host: ${byPlayer ? byPlayer.name : 'sistema'}`);
    }
};


// Flujo de datos ---------------------

// Variables
var matchId = null; // ID del partido actual
let scorer = null; // Variable para el jugador que anot車 el 迆ltimo gol


// Funci車n para obtener un nuevo match_id e inicializar el partido
function initializeMatchIdAndRegister() {
    fetch("http://localhost:3000/next-match-id")
        .then(response => {
            if (!response.ok) {
                throw new Error("Error al obtener el match_id");
            }
            return response.json();
        })
        .then(data => {
            matchId = data.next_match_id;
            console.log(`Nuevo partido inicializado con match_id: ${matchId}`);
            registerMatchStart(); // Registrar el inicio del partido
        })
        .catch(error => console.error("Error inicializando el match_id:", error));
}

// Funci車n para registrar el inicio del partido
function registerMatchStart() {
    const startTime = new Date().toISOString();
    fetch("http://localhost:3000/register-match", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ match_id: matchId, start_time: startTime }),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error("Error al registrar el partido");
            }
            console.log("Partido registrado correctamente en la base de datos.");
            isMatchActive = true; // Marca el partido como activo
        })
        .catch(error => console.error("Error registrando el partido:", error));
}

// Funci車n para registrar el final del partido
function endMatch() {
    const endTime = new Date().toISOString();
    fetch("http://localhost:3000/register-end-match", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ match_id: matchId, end_time: endTime }),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error("Error al registrar el fin del partido");
            }
            console.log("Fin del partido registrado correctamente.");
            isMatchActive = false; // Marca el partido como terminado
            matchId = null; // Reinicia el matchId
        })
        .catch(error => console.error("Error registrando el fin del partido:", error));
}

// Eventos del servidor de Haxball
room.onGameStart = function () {
    console.log("El partido ha comenzado.");
    if (!isMatchActive) {
        initializeMatchIdAndRegister(); // Obtener un nuevo match_id y registrar el inicio
    } else {
        console.log("El partido ya estaba activo, no se reinicia.");
    }
};

room.onGameStop = function () {
    console.log("El partido ha terminado.");
    console.log("Partido activo: ${isMatchActive}");

    if (isMatchActive) {
    isMatchActive = false;
        endMatch(); // Registrar el fin del partido
    } else {
        console.log("No hay partido activo para finalizar.");
    }
    mapChanged = false;
};





// Llamar al iniciar el servidor
// initializeMatchId();

// Variables de control
var playerPositions = []; // Almacena posiciones temporalmente.
var sendInterval = 60; // Cantidad de ticks por segundo (1 segundo = 60 ticks)
var lastSendTick = 0;
const tickInterval = 60; // Env赤a datos cada 60 ticks
let tickCounter = 0; // Contador de ticks
let lastTouchPlayer = null;

let previousPositions = {}; // Variables para almacenar posiciones previas

// Calcula la velocidad de un objeto
function calculateVelocity(current, previous, deltaTime) {
    return {
        velocity_x: (current.x - previous.x) / deltaTime,
        velocity_y: (current.y - previous.y) / deltaTime,
    };
}      


room.onGameTick = function () {
    if (!matchId) {
        console.warn("No hay un match_id activo. Ignorando datos.");
        return;
    }
    
    //Para saber si el partido esta activo
    //console.log(`Pelota en juego: ${isBallBeingPlayed}`);

    const players = room.getPlayerList();
    const ball = room.getBallPosition();
    const scores = room.getScores();
    const currentTime = scores ? scores.time : 0;
    const prev_time = -1;
    
    if(ball.x === 0 && ball.y === 0){
				afterGoal = false;
		}
		
		if(afterGoal){
		    if(!isBallBeingPlayed){
				    return;
		    }
    }
    
    if(ball.x === 0 && ball.y === 0 && !isBallBeingPlayed){
		    return;
    }

		
		isBallBeingPlayed = true;
		
    if (!players || !ball) {
        console.error("No se pudieron obtener jugadores o posici車n de la pelota.");
        return;
    }

    tickCounter++;

    if (tickCounter >= tickInterval) {
        tickCounter = 0;

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
        }).filter(player => {
            if (player.x == null || player.y == null || player.velocity_x == null || player.velocity_y == null) {
                console.error(`Error: La posici車n del jugador ${player.player_id} es nula.`);
                return false;
            }
            return true;
        });

        const ballData = (() => {
            const currentBall = { x: ball.x, y: ball.y, time: currentTime };
            const prevBall = previousPositions[0] || currentBall;
            const deltaTime = prevBall.time !== undefined ? currentTime - prevBall.time : 1;

            const velocity = calculateVelocity(currentBall, prevBall, deltaTime);
            previousPositions[0] = currentBall;

            if (currentBall.x == null || currentBall.y == null || velocity.velocity_x == null || velocity.velocity_y == null) {
                console.error('Error: La posici車n de la pelota o velocidades son nulas.');
                return null;
            }

            return {
                player_id: 0,
                player_name: "Ball",
                x: currentBall.x,
                y: currentBall.y,
                velocity_x: velocity.velocity_x,
                velocity_y: velocity.velocity_y
            };
        })();

        if (!ballData) {
            return;
        }

        const data = {
            time: currentTime,
            players: validPlayers,
            ball: ballData
        };

        playerPositions.push(data);
        sendDataToBackend(playerPositions);
        playerPositions = [];
    }

    let closestPlayer = null;
    let closestDistance = Infinity;

    players.forEach(player => {
        if (player.position) {
            let distance = Math.sqrt(
                Math.pow(player.position.x - ball.x, 2) +
                Math.pow(player.position.y - ball.y, 2)
            );

            if (distance < 50 && distance < closestDistance) {
                closestDistance = distance;
                closestPlayer = player;
            }
        }
    });

    if (closestPlayer) {
        lastTouchPlayer = closestPlayer;
    }
};


// Funci車n para enviar datos al servidor backend.
function sendDataToBackend(data) {
    if (!matchId) {
        console.error("No se puede enviar datos porque el match_id no est芍 definido.");
        return;
    }

    fetch("http://localhost:3000/save-positions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ match_id: matchId, positions: data }),
    }).catch(err => console.error("Error enviando posiciones al backend:", err));
}

// Captura 迆ltimo jugador en tocar la pelota
room.onPlayerBallTouch = function (player) {
    lastTouchPlayer = player;
};

// Captura goles
room.onTeamGoal = function (team) {
    let scoringTeam = (team === 1) ? "Red" : "Blue";
    isBallBeingPlayed = false;
    afterGoal = true;
    const scores = room.getScores();
    const goalTick = scores ? scores.time : 0;  // Obtener el tick del gol

    if (lastTouchPlayer) {
        console.log(`GOOOOL de ${lastTouchPlayer.name}, equipo ${scoringTeam} en el tick ${goalTick}`);
        fetch("http://localhost:3000/save-goal", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                match_id: matchId,
                player_id: lastTouchPlayer.id,
                player_name: lastTouchPlayer.name,
                equipo: scoringTeam,
                tick: goalTick  // Enviar el tick del gol
            }),
        }).catch(err => console.error("Error enviando gol:", err));
    }
};

// Captura victorias
room.onTeamVictory = function (scores) {
    isBallBeingPlayed = false;
    setTimeout(() => {
        // No cambiar autom芍ticamente a true
    }, 5000); // Mantener isBallBeingPlayed en falso durante 5 segundos

    console.log(`Victoria del equipo: ${scores.red > scores.blue ? 'Red' : 'Blue'}`);
};