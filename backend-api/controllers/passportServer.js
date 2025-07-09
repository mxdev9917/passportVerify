const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const errors = require('../utils/errors');
const db = require('../db/connection');
const createUpload = require('../utils/multer');
const { verify } = require('crypto');
const { rejects } = require('assert');

// Create multer upload middleware for /public/images
const uploadImage = createUpload('public/images');

// Helper function to format date strings as DD/MM/YYYY
function formatDateToDDMMYYYY(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const dd = String(date.getDate()).padStart(2, '0');
    const mm = String(date.getMonth() + 1).padStart(2, '0'); // Month is zero-based
    const yyyy = date.getFullYear();
    return `${dd}/${mm}/${yyyy}`;
}

exports.verifyPassport = (req, res, next) => {
    uploadImage.single("image")(req, res, async (err) => {
        if (err) {
            return res.status(400).json({ status: 400, message: err.message });
        }

        if (!req.file) {
            return res.status(400).json({ status: 400, message: 'No image uploaded' });
        }

        try {
            const filename = req.file.filename;
            await insertPathImg(filename);


            const imagePath = req.file.path;
            // Prepare image as form-data
            const form = new FormData();
            form.append('image', fs.createReadStream(imagePath));

            // Send image to MRZ API
            const response = await axios.post(
                'https://mrz.khatthaphone.com/evisa-ocr-api/v1/mrz',
                form,
                {
                    headers: form.getHeaders(),
                    timeout: 15000
                }
            );

            // Clean up uploaded image file
            fs.unlink(imagePath, (unlinkErr) => {
                if (unlinkErr) console.error("Error deleting uploaded file:", unlinkErr.message);
            });

            const passportNumber = response.data?.data?.passport?.passportNumber;

            if (!passportNumber) {
                return res.status(400).json({ message: "Passport number not found in MRZ response" });
            }

            const sql = `
                SELECT mrz, format, countryCode, passportNumber,
                       firstName, lastName, sex, birthDate, issueDate, expiryDate
                FROM Passport
                WHERE passportNumber = ?
            `;

            db.query(sql, [passportNumber], (error, results) => {
                if (error) {
                    console.error('DB Query Error:', error);
                    return errors.mapError(500, 'Error verifying passport', next);
                }

                // Format date fields before sending response
                const resultsFormatted = results.map(item => ({
                    ...item,
                    birthDate: formatDateToDDMMYYYY(item.birthDate),
                    issueDate: formatDateToDDMMYYYY(item.issueDate),
                    expiryDate: formatDateToDDMMYYYY(item.expiryDate),
                }));

                if (results > 0) {
                    return res.status(200).json({
                        message: 'Passport verified successfully',
                        valid: true,
                        data: resultsFormatted
                    });
                } else {
                    return res.status(200).json({
                        message: 'Passport not verified successfully',
                        valid: false,
                        data: response.data.data
                    });
                }
            });

        } catch (error) {
            console.error("MRZ API Error:", error.response?.data || error.message);
            return errors.mapError(500, 'Error verifying passport', next);
        }
    });
};

const insertPathImg = (filename) => {
    return new Promise((resolve, reject) => {
        const sql = `INSERT INTO Img_history (img_path) VALUES (?)`;
        db.query(sql, [filename], (error, results) => {
            if (error) {
                console.error("DB Error:", error);
                return reject(error);
            }
            // You can resolve with results or just resolve
            resolve(results);
        });
    });
};


exports.insertPassports = (req, res, next) => {
    try {
        const {
            mrz,
            format,
            countryCode,
            passportNumber,
            firstName,
            lastName,
            sex,
            birthDate,
            issueDate,
            expiryDate
        } = req.body;
        const sql = `
            INSERT INTO Passport (
                mrz, format, countryCode, passportNumber,
                firstName, lastName, sex, birthDate,
                issueDate, expiryDate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `;
        db.query(sql, [
            mrz, format, countryCode, passportNumber,
            firstName, lastName, sex, birthDate,
            issueDate, expiryDate
        ], (error) => {
            if (error) {
                console.error('DB Insert Error:', error);
                return errors.mapError(500, 'Error inserting passport', next);
            }

            return res.status(201).json({
                status: "201",
                message: "Successfully created passport record"
            });
        });
    } catch (error) {
        console.error('Unexpected Error:', error.message);
        errors.mapError(500, 'Internal server error', next);
    }
};

exports.deletePassportById = (req, res, next) => {
    const id = req.params.id;
    const sql = 'DELETE FROM Passport WHERE passport_id = ?';
    db.query(sql, [id], (error, results) => {
        if (error) {
            console.error('DB delete error:', error);
            return errors.mapError(500, 'Error deleting passport', next);
        }
        if (results.affectedRows === 0) {
            return res.status(404).json({ message: `Passport with id ${id} not found` });
        }
        return res.status(200).json({ message: `Passport with id ${id} deleted successfully` });
    });
};

