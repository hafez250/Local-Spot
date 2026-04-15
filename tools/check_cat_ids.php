<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$q1 = $m->query("SELECT c.category_id, cd.name FROM " . DB_PREFIX . "category c JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1 WHERE cd.name LIKE '%Starnine%'");
while ($r = $q1->fetch_assoc()) {
  echo "starnine: id=" . $r['category_id'] . " name=" . $r['name'] . PHP_EOL;
}

$q2 = $m->query("SELECT c.category_id, cd.name FROM " . DB_PREFIX . "category c LEFT JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1 WHERE c.category_id=61");
if ($q2 && $q2->num_rows) {
  $r = $q2->fetch_assoc();
  echo "id61: name=" . ($r['name'] ?? 'NULL') . PHP_EOL;
} else {
  echo "id61: not found" . PHP_EOL;
}