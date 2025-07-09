
exports.pathError = (req, res, next) => {
  const err = new Error(`Path ${req.originalUrl} not found in the server`);
  err.statusCode = 404;
  next(err);
}


exports.ApiError = (err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  const message = err.message || 'Error';
  res.status(statusCode).json({
    status: statusCode,
    message,
  });
};

exports.mapError = (status, msg, next) => {
  const error = new Error(msg);
  error.statusCode = status;
  next(error);
};
