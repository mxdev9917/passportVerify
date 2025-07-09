const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;
const routers =require('./routes/router')

// ✅ Load version from package.json
const { version } = require('./package.json');

app.use(express.json());

app.get('/', (req, res) => {
  res.send(`The passport verify backend system is working V.${version}`);
});
app.use('/api/', routers); 


app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}, Url: http://localhost:${PORT}`);
});
