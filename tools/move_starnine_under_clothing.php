<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$child = 61;
$parent = 492;

$m->begin_transaction();
try {
  $m->query("UPDATE " . DB_PREFIX . "category SET parent_id=$parent, status=1 WHERE category_id=$child");

  $checkStore = $m->query("SELECT 1 FROM " . DB_PREFIX . "category_to_store WHERE category_id=$child AND store_id=0 LIMIT 1");
  if (!$checkStore || !$checkStore->num_rows) {
    $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($child, 0)");
  }

  $m->query("DELETE FROM " . DB_PREFIX . "category_path WHERE category_id=$child");
  $parentPath = $m->query("SELECT path_id, level FROM " . DB_PREFIX . "category_path WHERE category_id=$parent ORDER BY level ASC");
  $level = 0;
  if ($parentPath) {
    while ($row = $parentPath->fetch_assoc()) {
      $pid = (int)$row['path_id'];
      $lvl = (int)$row['level'];
      $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($child, $pid, $lvl)");
      $level = $lvl + 1;
    }
  }
  $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($child, $child, $level)");

  $m->commit();
} catch (Throwable $e) {
  $m->rollback();
  throw $e;
}

$verify = $m->query("SELECT c.category_id,c.parent_id,cd.name FROM " . DB_PREFIX . "category c JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1 WHERE c.category_id IN (492,61,62) ORDER BY c.category_id");
while ($row = $verify->fetch_assoc()) {
  echo $row['category_id'] . ' parent=' . $row['parent_id'] . ' name=' . $row['name'] . PHP_EOL;
}