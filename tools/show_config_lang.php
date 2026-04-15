<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$r = $m->query("SELECT store_id,code,`key`,`value` FROM " . DB_PREFIX . "setting WHERE `key` LIKE 'config_language%' ORDER BY store_id,`key`");
while ($row = $r->fetch_assoc()) {
  echo $row['store_id'] . ' ' . $row['key'] . '=' . $row['value'] . PHP_EOL;
}