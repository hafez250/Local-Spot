<?php
require 'c:/xampp/htdocs/opencart1/config.php';

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$language_id = 1;
$res = $m->query("SELECT language_id FROM " . DB_PREFIX . "language WHERE code='en-gb' LIMIT 1");
if ($res && ($row = $res->fetch_assoc())) {
    $language_id = (int)$row['language_id'];
}

$asili_id = 0;
$idQuery = $m->query(
    "SELECT c.category_id
     FROM " . DB_PREFIX . "category c
     JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id
     WHERE c.parent_id=492 AND cd.language_id=$language_id AND LOWER(cd.name) LIKE 'asili%'
     ORDER BY c.category_id DESC
     LIMIT 1"
);
if ($idQuery && ($idRow = $idQuery->fetch_assoc())) {
    $asili_id = (int)$idRow['category_id'];
}

if ($asili_id <= 0) {
    echo "Asili not found\n";
    exit(1);
}

$sql = "SELECT DISTINCT p.product_id, p.image, pd.name
        FROM " . DB_PREFIX . "product p
        JOIN " . DB_PREFIX . "product_description pd ON pd.product_id=p.product_id AND pd.language_id=$language_id
        JOIN " . DB_PREFIX . "product_to_category p2c ON p2c.product_id=p.product_id
        JOIN " . DB_PREFIX . "category_path cp ON cp.category_id=p2c.category_id
        WHERE cp.path_id=$asili_id
        ORDER BY p.product_id DESC";

$q = $m->query($sql);
while ($r = $q->fetch_assoc()) {
    $img = (string)$r['image'];
    $full = DIR_IMAGE . html_entity_decode($img, ENT_QUOTES, 'UTF-8');
    $exists = ($img !== '' && is_file($full)) ? 'yes' : 'no';
    $size = ($exists === 'yes') ? (string)filesize($full) : '0';
    $dim = '0x0';
    if ($exists === 'yes') {
        $info = @getimagesize($full);
        if ($info) {
            $dim = $info[0] . 'x' . $info[1];
        }
    }
    echo $r['product_id'] . " | " . $r['name'] . " | " . $img . " | exists=" . $exists . " | size=" . $size . " | dim=" . $dim . PHP_EOL;
}

