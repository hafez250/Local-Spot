<?php
require __DIR__ . '/../config.php';

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$jackets_id = 496;
$svg_id = 62;

$q = $m->query("SELECT COUNT(DISTINCT p2c.product_id) c
  FROM " . DB_PREFIX . "product_to_category p2c
  JOIN " . DB_PREFIX . "category_path cp ON cp.category_id=p2c.category_id
  WHERE cp.path_id=$jackets_id");
$r = $q->fetch_assoc();
echo "jackets_desc_count=" . (int)$r['c'] . PHP_EOL;

$q2 = $m->query("SELECT COUNT(DISTINCT p2c.product_id) c
  FROM " . DB_PREFIX . "product_to_category p2c
  JOIN " . DB_PREFIX . "category_path cp ON cp.category_id=p2c.category_id
  WHERE cp.path_id=$svg_id");
$r2 = $q2->fetch_assoc();
echo "svg_desc_count=" . (int)$r2['c'] . PHP_EOL;

$q3 = $m->query("SELECT p.product_id, p.sku, pd.name
  FROM " . DB_PREFIX . "product p
  JOIN " . DB_PREFIX . "product_description pd ON pd.product_id=p.product_id AND pd.language_id=1
  JOIN " . DB_PREFIX . "product_to_category p2c ON p2c.product_id=p.product_id
  JOIN " . DB_PREFIX . "category_path cp ON cp.category_id=p2c.category_id
  WHERE cp.path_id=$jackets_id
  GROUP BY p.product_id
  ORDER BY p.product_id DESC
  LIMIT 20");

while ($row = $q3->fetch_assoc()) {
    echo $row['product_id'] . '|' . $row['sku'] . '|' . $row['name'] . PHP_EOL;
}

$q4 = $m->query("SELECT COUNT(DISTINCT p.product_id) c
  FROM " . DB_PREFIX . "product p
  JOIN " . DB_PREFIX . "product_description pd ON pd.product_id=p.product_id AND pd.language_id=1
  JOIN " . DB_PREFIX . "product_to_category p2c ON p2c.product_id=p.product_id
  JOIN " . DB_PREFIX . "category_path cp ON cp.category_id=p2c.category_id
  WHERE cp.path_id=$svg_id AND LOWER(pd.name) LIKE '%jacket%'");
$r4 = $q4->fetch_assoc();
echo "svg_name_like_jacket=" . (int)$r4['c'] . PHP_EOL;

$q5 = $m->query("SELECT `value` FROM `" . DB_PREFIX . "setting` WHERE `store_id`=0 AND `key`='config_pagination' LIMIT 1");
if ($q5 && ($r5 = $q5->fetch_assoc())) {
    echo "config_pagination=" . (int)$r5['value'] . PHP_EOL;
}

$m->close();
