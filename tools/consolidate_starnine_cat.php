<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

// Move products from duplicate starnine category to id 61 and drop duplicate category links
$dup = 491;
$main = 61;

$m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) SELECT product_id, $main FROM " . DB_PREFIX . "product_to_category WHERE category_id=$dup");
$m->query("DELETE FROM " . DB_PREFIX . "product_to_category WHERE category_id=$dup");

$tables = [
  DB_PREFIX . 'category',
  DB_PREFIX . 'category_description',
  DB_PREFIX . 'category_path',
  DB_PREFIX . 'category_to_store',
  DB_PREFIX . 'category_to_layout',
  DB_PREFIX . 'category_filter'
];
foreach ($tables as $t) {
  $chk = $m->query("SHOW TABLES LIKE '" . $m->real_escape_string($t) . "'");
  if (!$chk || !$chk->num_rows) continue;
  if ($t === DB_PREFIX . 'category_path') {
    $m->query("DELETE FROM `$t` WHERE category_id=$dup OR path_id=$dup");
  } else {
    $m->query("DELETE FROM `$t` WHERE category_id=$dup");
  }
}

$r = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$main");
$row = $r->fetch_assoc();
echo "cat61_products=" . $row['c'] . PHP_EOL;
