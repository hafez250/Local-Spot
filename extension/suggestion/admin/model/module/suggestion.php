<?php
namespace Opencart\Admin\Model\Extension\Suggestion\Module;

class Suggestion extends \Opencart\System\Engine\Model {

    public function install(): void {
        // Create table
        // $this->db->query("CREATE TABLE IF NOT EXISTS `" . DB_PREFIX . "suggestion` ...");
    }

    public function uninstall(): void {
        // Delete settings
        $this->db->query("DELETE FROM `" . DB_PREFIX . "setting` WHERE code = 'module_suggestion'");
        
        // Delete table
        // $this->db->query("DROP TABLE IF EXISTS `" . DB_PREFIX . "suggestion`");
    }

    public function getData(): array {
        $query = $this->db->query("SELECT * FROM " . DB_PREFIX . "setting WHERE code = 'module_suggestion'");
        return $query->rows;
    }

}