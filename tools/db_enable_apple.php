<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$m->query("UPDATE " . DB_PREFIX . "product SET status=1, quantity=100 WHERE sku LIKE 'P%'");
$r = $m->query("SELECT COUNT(*) c, SUM(status=1) s FROM " . DB_PREFIX . "product WHERE sku LIKE 'P%'");
$row = $r->fetch_assoc();
echo "total=" . $row['c'] . " status1=" . $row['s'] . PHP_EOL;