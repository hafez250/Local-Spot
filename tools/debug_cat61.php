<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$r = $m->query("SELECT category_id,parent_id,status FROM " . DB_PREFIX . "category WHERE category_id=61");
if ($r && $r->num_rows) {
  $row = $r->fetch_assoc();
  echo "category61 status=".$row['status']." parent=".$row['parent_id'].PHP_EOL;
} else {
  echo "category61 missing".PHP_EOL;
}

$r2 = $m->query("SELECT cd.language_id,l.code,cd.name FROM " . DB_PREFIX . "category_description cd LEFT JOIN " . DB_PREFIX . "language l ON l.language_id=cd.language_id WHERE cd.category_id=61");
while ($row = $r2->fetch_assoc()) {
  echo "desc lang=".$row['language_id']." code=".$row['code']." name=".$row['name'].PHP_EOL;
}

$r3 = $m->query("SELECT language_id,code,status FROM " . DB_PREFIX . "language");
while ($row = $r3->fetch_assoc()) {
  echo "lang ".$row['language_id']." ".$row['code']." status=".$row['status'].PHP_EOL;
}