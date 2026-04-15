<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$r = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=62");
$row = $r->fetch_assoc();
echo "svg_category_products=" . $row['c'] . PHP_EOL;

$q = $m->query("SELECT c.category_id, cd.name
  FROM " . DB_PREFIX . "category c
  JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1
  WHERE c.parent_id=62
  ORDER BY cd.name");
if ($q) {
  while ($r2 = $q->fetch_assoc()) {
    $cid = (int)$r2['category_id'];
    $cntQ = $m->query("SELECT COUNT(DISTINCT product_id) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$cid");
    $cnt = 0;
    if ($cntQ && ($cr = $cntQ->fetch_assoc())) {
      $cnt = (int)$cr['c'];
    }
    echo "sub=" . $r2['name'] . " id=" . $cid . " products=" . $cnt . PHP_EOL;
  }
}
