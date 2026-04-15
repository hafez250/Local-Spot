<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$category_id = 61;
$store_id = 0;

$exists = $m->query("SELECT 1 FROM " . DB_PREFIX . "category_to_store WHERE category_id=$category_id AND store_id=$store_id LIMIT 1");
if (!$exists || !$exists->num_rows) {
  $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($category_id, $store_id)");
  echo "category_to_store inserted\n";
} else {
  echo "category_to_store already exists\n";
}

$check = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "category_to_store WHERE category_id=$category_id");
$row = $check->fetch_assoc();
echo "store_links=" . $row['c'] . PHP_EOL;