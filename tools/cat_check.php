<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$sql = "SELECT cd.name AS category_name, COUNT(DISTINCT p.product_id) AS cnt
FROM " . DB_PREFIX . "product p
JOIN " . DB_PREFIX . "product_to_category p2c ON p.product_id = p2c.product_id
JOIN " . DB_PREFIX . "category_description cd ON cd.category_id = p2c.category_id AND cd.language_id = 1
GROUP BY cd.name
ORDER BY cnt DESC
LIMIT 50";
$r = $m->query($sql);
while ($row = $r->fetch_assoc()) {
  echo $row['category_name'] . ':' . $row['cnt'] . PHP_EOL;
}