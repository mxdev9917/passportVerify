const multer = require('multer');
const path = require('path');


const createUpload = (folderPath) => {
  const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, folderPath),
    filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
  });

  const upload = multer({
    storage,
    limits: { fileSize: 2 * 1024 * 1024 }, // 2MB
    fileFilter: (req, file, cb) => {
      const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
      if (!allowedTypes.includes(file.mimetype)) {
        return cb(new Error('Only JPG, JPEG, and PNG images are allowed'));
      }
      cb(null, true);
    }
  });

  return upload;
};

module.exports = createUpload;
