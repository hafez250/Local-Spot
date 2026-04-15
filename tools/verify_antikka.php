<?php
require 'c:/xampp/htdocs/opencart1/config.php';

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$language_id = 1;
$res = $m->query("SELECT language_id FROM " . DB_PREFIX . "language WHERE code='en-gb' LIMIT 1");
if ($res && $row = $res->fetch_assoc()) {
    $language_id = (int)$row['language_id'];
}

$idSql = "SELECT c.category_id
          FROM " . DB_PREFIX . "category c
          JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=$language_id
          WHERE cd.name='Antikka'
          LIMIT 1";
$idRes = $m->query($idSql);
$antikkaId = ($idRes && ($idRow = $idRes->fetch_assoc())) ? (int)$idRow['category_id'] : 0;

$direct = 0;
$tree = 0;

if ($antikkaId > 0) {
    $q1 = $m->query("SELECT COUNT(DISTINCT product_id) c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$antikkaId");
    if ($q1 && ($r1 = $q1->fetch_assoc())) {
        $direct = (int)$r1['c'];
    }

    $q2 = $m->query("SELECT COUNT(DISTINCT p2c.product_id) c
                     FROM " . DB_PREFIX . "product_to_category p2c
                     JOIN " . DB_PREFIX . "category_path cp ON cp.category_id=p2c.category_id
                     WHERE cp.path_id=$antikkaId");
    if ($q2 && ($r2 = $q2->fetch_assoc())) {
        $tree = (int)$r2['c'];
    }
}

echo 'AntikkaCategoryID=' . $antikkaId . PHP_EOL;
echo 'AntikkaDirectProducts=' . $direct . PHP_EOL;
echo 'AntikkaTreeProducts=' . $tree . PHP_EOL;
