/**
 * Prepares the bundled 'www' web directory for the Capacitor native app.
 */
const fs = require('fs');
const path = require('path');

const wwwDir = path.join(__dirname, 'www');
if (!fs.existsSync(wwwDir)) {
  fs.mkdirSync(wwwDir, { recursive: true });
}

// Read the PAGE string directly from serve.py
const servePyPath = path.join(__dirname, '..', 'cardcenter', 'serve.py');
const servePyContent = fs.readFileSync(servePyPath, 'utf8');

const match = servePyContent.match(/PAGE\s*=\s*"""([\s\S]*?)"""/);
if (!match) {
  console.error("Could not extract PAGE template from serve.py");
  process.exit(1);
}

let htmlContent = match[1];

// Write to www/index.html
fs.writeFileSync(path.join(wwwDir, 'index.html'), htmlContent, 'utf8');
console.log("Successfully prepared mobile web bundle at mobile/www/index.html");
