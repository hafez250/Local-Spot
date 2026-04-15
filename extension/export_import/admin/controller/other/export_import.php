<?php
namespace Opencart\Admin\Controller\Extension\ExportImport\Other;

class ExportImport extends \Opencart\System\Engine\Controller {

	protected string $method_separator;


	public function __construct(\Opencart\System\Engine\Registry $registry) {
		parent::__construct($registry);
		$this->method_separator = version_compare(VERSION,'4.0.2.0','>=') ? '.' : '|';
	}


	public function index(): void {
		$this->load->language('extension/export_import/other/export_import');

		$this->document->setTitle($this->language->get('heading_title'));

		$data['breadcrumbs'] = [];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_home'),
			'href' => $this->url->link('common/dashboard', 'user_token=' . $this->session->data['user_token'])
		];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_extension'),
			'href' => $this->url->link('marketplace/extension', 'user_token=' . $this->session->data['user_token'] . '&type=other')
		];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('heading_title'),
			'href' => $this->url->link('extension/export_import/other/export_import', 'user_token=' . $this->session->data['user_token'])
		];

		$data['save'] = $this->url->link('extension/export_import/other/export_import'.$this->method_separator.'save', 'user_token=' . $this->session->data['user_token']);
		$data['back'] = $this->url->link('marketplace/extension', 'user_token=' . $this->session->data['user_token'] . '&type=other');

		$data['other_export_import_status'] = $this->config->get('other_export_import_status');

		$data['success'] = '';
		if (!empty($this->session->data['other_export_import_success'])) {
			$data['success'] = $this->session->data['other_export_import_success'];
			unset($this->session->data['other_export_import_success']);
		}

		$data['header'] = $this->load->controller('common/header');
		$data['column_left'] = $this->load->controller('common/column_left');
		$data['footer'] = $this->load->controller('common/footer');

		$this->response->setOutput($this->load->view('extension/export_import/other/export_import', $data));
	}


	public function save(): void {
		$this->load->language('extension/export_import/other/export_import');

		$json = [];

		if (!$this->user->hasPermission('modify', 'extension/export_import/other/export_import')) {
			$json['error'] = $this->language->get('error_permission');
		}

		if (!$json) {
			$this->load->model('setting/setting');

			$this->model_setting_setting->editSetting('other_export_import', $this->request->post);

			$json['redirect'] = str_replace('&amp;','&',$this->url->link('extension/export_import/other/export_import', 'user_token=' . $this->session->data['user_token']));
			$this->session->data['other_export_import_success'] = $this->language->get('text_success');
		}

		$this->response->addHeader('Content-Type: application/json');
		$this->response->setOutput(json_encode($json));
	}


	public function install(): void {
		// add events
		$this->load->model('setting/event');
		if (version_compare(VERSION,'4.0.1.0','>=')) {
			$data = [
				'code'        => 'other_export_import',
				'description' => '',
				'trigger'     => 'admin/view/common/column_left/before',
				'action'      => 'extension/export_import/other/export_import'.$this->method_separator.'eventViewCommonColumnLeftBefore',
				'status'      => true,
				'sort_order'  => 0
			];
			$this->model_setting_event->addEvent($data);
		} else {
			$this->model_setting_event->addEvent('other_export_import','','admin/view/common/column_left/before','extension/export_import/other/export_import'.$this->method_separator.'eventViewCommonColumnLeftBefore');
		}

		// add access rights
		$this->addAccessRights();
	}


	public function uninstall(): void {
		// remove events
		$this->load->model('setting/event');
		$this->model_setting_event->deleteEventByCode('other_export_import');

		// remove access rights
		$this->removeAccessRights();
	}


	public function eventViewCommonColumnLeftBefore(&$route, &$data, &$code) {
		if (!$this->config->get('other_export_import_status')) {
			return null;
		}

		$this->load->language('common/column_left');
		$text_maintenance = $this->language->get('text_maintenance');
		$this->load->language('extension/export_import/other/export_import');
		$text_export_import = $this->language->get('menu_export_import');

		foreach ($data['menus'] as $key0=>$menu) {
			if ($menu['id'] == 'menu-system') {
				foreach ($menu['children'] as $key1=>$system_child) {
					if ($system_child['name']==$text_maintenance) {
						// add Export/Import to System > Maintenance menu
						$data['menus'][$key0]['children'][$key1]['children'][] = [
							'name'     => $text_export_import,
							'href'     => $this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token']),
							'children' => []
						];
						return null;
					}
				}
				// Add Export/Import to System menu
				$data['menus'][$key0]['children'][] = [
					'name'	   => $text_export_import,
					'href'     => $this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token']),
					'children' => []
				];
				return null;
			}
		}

		// Add Export/Import as a top-level menu
		$data['menus'][] = [
			'id'       => 'menu-export-impport',
			'icon'     => 'fas fa-cog',
			'name'     => $text_export_import,
			'href'     => $this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token']),
			'children' => []
		];
		return null;
	}


	protected function addAccessRights() {
		$this->load->model('user/user_group');
		$this->model_user_user_group->addPermission($this->user->getGroupId(), 'access', 'extension/export_import/tool/export_import');
		$this->model_user_user_group->addPermission($this->user->getGroupId(), 'modify', 'extension/export_import/tool/export_import');
	}


	protected function removeAccessRights() {
		$this->load->model('user/user_group');
		$this->load->model('extension/export_import/other/export_import');
		$user_groups = $this->model_user_user_group->getUserGroups();
		foreach ($user_groups as $user_group) {
			$user_group_id = $user_group['user_group_id'];
			$this->model_extension_export_import_other_export_import->removePermission($user_group_id, 'access', 'extension/export_import/tool/export_import');
			$this->model_extension_export_import_other_export_import->removePermission($user_group_id, 'modify', 'extension/export_import/tool/export_import');
			$this->model_extension_export_import_other_export_import->removePermission($user_group_id, 'access', 'extension/export_import/other/export_import');
			$this->model_extension_export_import_other_export_import->removePermission($user_group_id, 'modify', 'extension/export_import/other/export_import');
		}
	}
}
?>