<?php
namespace Opencart\Admin\Model\Extension\ExportImport\Other;

class ExportImport extends \Opencart\System\Engine\Model {

	public function addPermission(int $user_group_id, string $type, string $route): void {
		$this->load->model('user/user_group');
		$this->model_user_user_group->addPermission($user_group_id,$type,$route);
	}

	public function removePermission(int $user_group_id, string $type, string $route): void {
		// the standard OpenCart 4.0.0.0 removePermission is buggy, hence we use our own one!
		$user_group_query = $this->db->query("SELECT DISTINCT * FROM `" . DB_PREFIX . "user_group` WHERE `user_group_id` = '" . (int)$user_group_id . "'");
		if ($user_group_query->num_rows) {
			$data = json_decode($user_group_query->row['permission'], true);
			if (!empty($data[$type])) {
				$data[$type] = array_diff($data[$type], [$route]);
				$this->db->query("UPDATE `" . DB_PREFIX . "user_group` SET `permission` = '" . $this->db->escape(json_encode($data)) . "' WHERE `user_group_id` = '" . (int)$user_group_id . "'");
			}
		}
	}
}
?>