const express = require("express");
const cors = require("cors");
const { Client } = require("pg");
const dotenv = require("dotenv");

// Cargar variables de entorno desde el archivo .env
dotenv.config();

const app = express();

// Configurar CORS para permitir solicitudes desde Haxball
app.use(cors({
    origin: "https://www.haxball.com",
}));
app.use(express.json());

// Configuración de la base de datos PostgreSQL usando variables de entorno
const client = new Client({
    user: process.env.DB_USERNAME,
    host: process.env.DB_HOST,
    database: process.env.DB_DATABASE,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

client.connect((err) => {
    if (err) {
        console.error("Error al conectar a PostgreSQL:", err);
    } else {
        console.log("Conexión a PostgreSQL exitosa.");
    }
});

// Endpoint para obtener el próximo match_id
app.get("/next-match-id", async (req, res) => {
    try {
        const result = await client.query("SELECT COALESCE(MAX(match_id), 0) + 1 AS next_match_id FROM Partidos");
        res.json({ next_match_id: result.rows[0].next_match_id });
    } catch (err) {
        res.status(500).send("Error al obtener el próximo match_id.");
    }
});

// Endpoint para registrar el inicio de un partido
app.post("/register-match", async (req, res) => {
    const { match_id, start_time } = req.body;
    try {
        await client.query("INSERT INTO Partidos (match_id, start_time) VALUES ($1, $2)", [match_id, start_time]);
        res.send("Partido registrado correctamente.");
    } catch (err) {
        res.status(500).send("Error al registrar el partido.");
    }
});

// Endpoint para registrar el fin de un partido
app.post("/register-end-match", async (req, res) => {
    const { match_id, end_time } = req.body;
    try {
        await client.query("UPDATE Partidos SET end_time = $1 WHERE match_id = $2", [end_time, match_id]);
        res.send("Fin del partido registrado correctamente.");
    } catch (err) {
        res.status(500).send("Error al registrar el fin del partido.");
    }
});

// Endpoint para guardar las posiciones de los jugadores y el balón
app.post("/save-positions", async (req, res) => {
    const { match_id, positions } = req.body;

    // Mostrar el contenido de 'positions' y 'players' en la consola del servidor
    console.log("Positions received:", positions);
    positions.forEach(position => {
        console.log("Players:", position.players);
    });

    const query = `
        INSERT INTO posiciones
        (match_id, player_id, x, y, velocity_x, velocity_y, time, team) 
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `;

    try {
        for (const position of positions) {
            // Insertar datos de jugadores
            for (const player of position.players) {
                // Obtener el ID del jugador desde la base de datos
                const playerResult = await client.query(
                    "SELECT player_id FROM Jugadores WHERE player_name = $1",
                    [player.player_name]
                );

                if (playerResult.rows.length > 0) {
                    const playerId = playerResult.rows[0].player_id;
                    const value = [
                        match_id,
                        playerId,
                        player.x ?? 0,
                        player.y ?? 0,
                        player.velocity_x ?? 0,
                        player.velocity_y ?? 0,
                        position.time,
                        player.team
                    ];

                    await client.query(query, value);
                } else {
                    console.error(`Error: No se encontró el jugador ${player.player_name} en la base de datos.`);
                }
            }

            // Insertar datos de la pelota
            const ballValue = [
                match_id,
                position.ball.player_id,
                position.ball.x ?? 0,
                position.ball.y ?? 0,
                position.ball.velocity_x ?? 0,
                position.ball.velocity_y ?? 0,
                position.time,
                0
            ];

            await client.query(query, ballValue);
        }
        res.status(200).send("Datos guardados correctamente.");
    } catch (err) {
        console.error("Error al guardar los datos:", err);
        res.status(500).send("Error al guardar los datos.");
    }
});




// Endpoint para guardar goles
app.post("/save-goal", async (req, res) => {
    const { match_id, player_name, equipo, tick } = req.body;  // Obtener el tick del gol

    try {
        // Obtener el ID del jugador desde la base de datos
        const playerResult = await client.query(
            "SELECT player_id FROM Jugadores WHERE player_name = $1",
            [player_name]
        );

        if (playerResult.rows.length > 0) {
            const playerId = playerResult.rows[0].player_id;

            await client.query(`
                INSERT INTO Goles (match_id, player_id, player_name, equipo, tick)
                VALUES ($1, $2, $3, $4, $5)
            `, [match_id, playerId, player_name, equipo, tick]);  // Insertar el tick del gol

            res.send("Gol registrado correctamente.");
        } else {
            console.error(`Error: No se encontró el jugador ${player_name} en la base de datos.`);
            res.status(404).send("Jugador no encontrado.");
        }
    } catch (err) {
        console.error("Error al registrar el gol:", err);
        res.status(500).send("Error al registrar el gol.");
    }
});

// Endpoint para Registrar Jugadores (Incluida la Pelota)
app.post("/register-player", async (req, res) => {
    const { player_id, player_name } = req.body;
    try {
        await client.query(
            "INSERT INTO Jugadores (player_id, player_name) VALUES ($1, $2) ON CONFLICT (player_id) DO NOTHING",
            [player_id, player_name]
        );
        res.send("Jugador registrado correctamente.");
    } catch (err) {
        console.error("Error registrando el jugador:", err);
        res.status(500).send("Error al registrar el jugador.");
    }
});

app.post("/get-or-create-unique-player", async (req, res) => {
    const { player_name } = req.body;
    try {
        // 1. Verifica si el jugador ya existe
        const existingPlayer = await client.query(
            "SELECT player_id FROM Jugadores WHERE player_name = $1",
            [player_name]
        );

        if (existingPlayer.rows.length > 0) {
            // Jugador ya está registrado, devuelve el player_id existente
            return res.json({ player_id: existingPlayer.rows[0].player_id });
        }

        // 2. Jugador no registrado: genera un ID único que no esté en la tabla
        const usedIdsResult = await client.query("SELECT player_id FROM Jugadores");
        const usedIds = usedIdsResult.rows.map(row => row.player_id);

        // Encuentra el menor ID disponible (puedes usar otra lógica si prefieres)
        let newPlayerId = 1;
        while (usedIds.includes(newPlayerId)) {
            newPlayerId++;
        }

        // 3. Registra el nuevo jugador con el ID único
        await client.query(
            "INSERT INTO Jugadores (player_id, player_name) VALUES ($1, $2)",
            [newPlayerId, player_name]
        );

        res.json({ player_id: newPlayerId });
    } catch (err) {
        console.error("Error al obtener o crear jugador:", err);
        res.status(500).send("Error al obtener o crear jugador.");
    }
});


app.post("/get-or-create-player", async (req, res) => {
    const { player_name } = req.body;
    try {
        // Verifica si el jugador ya existe
        const result = await client.query(
            "SELECT player_id FROM Jugadores WHERE player_name = $1",
            [player_name]
        );

        let player_id;

        if (result.rows.length > 0) {
            // Jugador ya registrado
            player_id = result.rows[0].player_id;
        } else {
            // Jugador no registrado, crear nuevo registro
            const insertResult = await client.query(
                "INSERT INTO Jugadores (player_name) VALUES ($1) RETURNING player_id",
                [player_name]
            );
            player_id = insertResult.rows[0].player_id;
        }

        res.json({ player_id });
    } catch (err) {
        console.error("Error al obtener o crear jugador:", err);
        res.status(500).send("Error al obtener o crear jugador.");
    }
});




// Iniciar servidor
app.listen(3000, () => {
    console.log("Servidor corriendo en http://localhost:3000");
});