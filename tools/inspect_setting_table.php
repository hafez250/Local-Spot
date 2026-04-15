<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$r = $m->query("SHOW COLUMNS FROM " . DB_PREFIX . "setting");
while ($row = $r->fetch_assoc()) { echo $row['Field'] . PHP_EOL; }

echo "---sample---" . PHP_EOL;
$r2 = $m->query("SELECT * FROM " . DB_PREFIX . "setting LIMIT 20");
while ($row = $r2->fetch_assoc()) {
  echo json_encode($row, JSON_UNESCAPED_UNICODE) . PHP_EOL;
}