<?php
namespace Opencart\Catalog\Model\Extension\Suggestion\Module;

class Suggestion extends \Opencart\System\Engine\Model {

    public function getData(): array {
        $query = $this->db->query("SELECT * FROM " . DB_PREFIX . "setting WHERE code = 'module_suggestion'");
        return $query->rows;
    }

    public function getModuleData(): array {
        // Add your custom data retrieval logic here
        return [];
    }
}