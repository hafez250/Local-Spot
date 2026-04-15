<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$parent = 492;
$children = [61, 62];

$childList = implode(',', array_map('intval', $children));

// Add parent-category links for all products currently in child brand categories.
$m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) SELECT DISTINCT product_id, $parent FROM " . DB_PREFIX . "product_to_category WHERE category_id IN ($childList)");

$r = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$parent");
$row = $r->fetch_assoc();

echo "clothing_products=" . $row['c'] . PHP_EOL;