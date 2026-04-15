<?php
require __DIR__ . '/../config.php';

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($m->connect_error) {
    fwrite(STDERR, "DB connect error: {$m->connect_error}" . PHP_EOL);
    exit(1);
}

$cat = 493;
$parent = 492;

$q1 = $m->query("SELECT c.category_id, c.parent_id, cd.name
  FROM " . DB_PREFIX . "category c
  JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1
  WHERE c.category_id=$cat");
if ($q1 && ($r = $q1->fetch_assoc())) {
    echo "category=" . $r['category_id'] . " parent=" . $r['parent_id'] . " name=" . $r['name'] . PHP_EOL;
}

$q2 = $m->query("SELECT COUNT(DISTINCT product_id) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$cat");
if ($q2 && ($r = $q2->fetch_assoc())) {
    echo "seemly_products=" . $r['c'] . PHP_EOL;
}

$q3 = $m->query("SELECT COUNT(DISTINCT product_id) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$parent");
if ($q3 && ($r = $q3->fetch_assoc())) {
    echo "clothing_products=" . $r['c'] . PHP_EOL;
}

$q4 = $m->query("SELECT c.category_id, c.parent_id, cd.name
  FROM " . DB_PREFIX . "category c
  JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=1
  WHERE c.parent_id=$cat
  ORDER BY cd.name");
if ($q4) {
    while ($r = $q4->fetch_assoc()) {
        $cid = (int)$r['category_id'];
        $cntQ = $m->query("SELECT COUNT(DISTINCT product_id) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$cid");
        $cnt = 0;
        if ($cntQ && ($cr = $cntQ->fetch_assoc())) {
            $cnt = (int)$cr['c'];
        }
        echo "sub=" . $r['name'] . " id=" . $cid . " products=" . $cnt . PHP_EOL;
    }
}

$m->close();
