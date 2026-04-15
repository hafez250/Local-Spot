<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$r = $m->query("SELECT c.category_id, cd.name FROM " . DB_PREFIX . "category c JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1 WHERE LOWER(cd.name) LIKE '%svg%'");
while ($row = $r->fetch_assoc()) {
  echo $row['category_id'] . ':' . $row['name'] . PHP_EOL;
}