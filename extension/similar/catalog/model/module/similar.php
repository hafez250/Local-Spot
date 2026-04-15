<?php
namespace Opencart\Catalog\Model\Extension\Similar\Module;

class Similar extends \Opencart\System\Engine\Model {

    public function getData(): array {
        $query = $this->db->query("SELECT * FROM " . DB_PREFIX . "setting WHERE code = 'module_similar'");
        return $query->rows;
    }

    public function getModuleData(): array {
        // Add your custom data retrieval logic here
        return [];
    }
}