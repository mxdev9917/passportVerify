const express = require('express');
const { insertPassports,deletePassportById,verifyPassport } = require('../controllers/passportServer');

const Router = express.Router();
Router.post('/verify', verifyPassport); 
Router.post('/insert', insertPassports); 
Router.delete('/delete/:id', deletePassportById);

module.exports = Router;