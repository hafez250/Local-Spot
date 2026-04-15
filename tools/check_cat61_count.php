<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$r = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=61");
$row = $r->fetch_assoc();
echo "cat61_products=" . $row['c'] . PHP_EOL;