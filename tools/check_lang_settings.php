<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
foreach (['config_language','config_language_id'] as $k) {
  $kEsc = $m->real_escape_string($k);
  $r = $m->query("SELECT store_id,`value` FROM " . DB_PREFIX . "setting WHERE `key`='$kEsc' ORDER BY store_id");
  while ($row = $r->fetch_assoc()) {
    echo $k . ' store=' . $row['store_id'] . ' value=' . $row['value'] . PHP_EOL;
  }
}