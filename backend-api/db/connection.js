// const encrypt = require('../utils/encrypt');
const { createPool } = require('mysql2');
const { config } = require('dotenv');
config({ path: './config.env' });

const pool = createPool({
    connectionLimit: 10,
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_DATABASE,
    connectTimeout: parseInt(process.env.DB_CONNECTTIMEOUT) || 10000,
    waitForConnections: true,
    queueLimit: 0
});

pool.getConnection((err, connection) => {
    if (err) {
        console.error('Database connection error:', err.message);
        return;
    }
    console.log('Connected to MySQL database');
});

module.exports = pool;
