<?php
namespace Opencart\Admin\Controller\Extension\ExportImport\Tool;

class ExportImport extends \Opencart\System\Engine\Controller {

	private $error = array();

	protected string $method_separator;


	public function __construct(\Opencart\System\Engine\Registry $registry) {
		parent::__construct($registry);
		$this->method_separator = version_compare(VERSION,'4.0.2.0','>=') ? '.' : '|';
	}



	public function index() {
		if (!$this->config->get('other_export_import_status')) {
			$url = $this->url->link('error/not_found','user_token='.$this->session->data['user_token'] );
			$this->response->redirect( $url );
		}
		$this->load->language('extension/export_import/tool/export_import');
		$this->document->setTitle($this->language->get('heading_title'));
		$this->load->model('extension/export_import/tool/export_import');
		$this->getForm();
	}


	public function upload() {
		$this->load->language('extension/export_import/tool/export_import');
		$this->document->setTitle($this->language->get('heading_title'));
		$this->load->model('extension/export_import/tool/export_import');
		if (($this->request->server['REQUEST_METHOD'] == 'POST') && ($this->validateUploadForm())) {
			if ((isset( $this->request->files['upload'] )) && (is_uploaded_file($this->request->files['upload']['tmp_name']))) {
				$file = $this->request->files['upload']['tmp_name'];
				$incremental = ($this->request->post['incremental']) ? true : false;
				if ($this->model_extension_export_import_tool_export_import->upload($file,$this->request->post['incremental'])==true) {
					$this->session->data['success'] = $this->language->get('text_success');
					$this->response->redirect($this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token'] ));
				}
				else {
					$this->session->data['warning'] = $this->language->get('error_upload');
					$href = $this->url->link( 'tool/log', 'user_token='.$this->session->data['user_token'] );
					$this->session->data['warning'] .= "<br />\n".str_replace('%1',$href,$this->language->get( 'text_log_details_3_x' ));
					$this->response->redirect($this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token']));
				}
			}
		}

		$this->getForm();
	}


	protected function return_bytes($val)
	{
		$val = trim($val);
	
		switch (strtolower(substr($val, -1)))
		{
			case 'm': $val = (int)substr($val, 0, -1) * 1048576; break;
			case 'k': $val = (int)substr($val, 0, -1) * 1024; break;
			case 'g': $val = (int)substr($val, 0, -1) * 1073741824; break;
			case 'b':
				switch (strtolower(substr($val, -2, 1)))
				{
					case 'm': $val = (int)substr($val, 0, -2) * 1048576; break;
					case 'k': $val = (int)substr($val, 0, -2) * 1024; break;
					case 'g': $val = (int)substr($val, 0, -2) * 1073741824; break;
					default : break;
				} break;
			default: break;
		}
		return $val;
	}


	public function download() {
		$this->load->language( 'extension/export_import/tool/export_import' );
		$this->document->setTitle($this->language->get('heading_title'));
		$this->load->model( 'extension/export_import/tool/export_import' );
		if (($this->request->server['REQUEST_METHOD'] == 'POST') && $this->validateDownloadForm()) {
			$export_type = $this->request->post['export_type'];
			switch ($export_type) {
				case 'c':
				case 'p':
				case 'u':
					$min = null;
					if (isset( $this->request->post['min'] ) && ($this->request->post['min']!='')) {
						$min = $this->request->post['min'];
					}
					$max = null;
					if (isset( $this->request->post['max'] ) && ($this->request->post['max']!='')) {
						$max = $this->request->post['max'];
					}
					if (($min==null) || ($max==null)) {
						$this->model_extension_export_import_tool_export_import->download($export_type, null, null, null, null);
					} else if ($this->request->post['range_type'] == 'id') {
						$this->model_extension_export_import_tool_export_import->download($export_type, null, null, $min, $max);
					} else {
						$this->model_extension_export_import_tool_export_import->download($export_type, $min*($max-1-1), $min, null, null);
					}
					break;
				case 'o':
					$this->model_extension_export_import_tool_export_import->download('o', null, null, null, null);
					break;
				case 'a':
					$this->model_extension_export_import_tool_export_import->download('a', null, null, null, null);
					break;
				case 'f':
					$this->model_extension_export_import_tool_export_import->download('f', null, null, null, null);
					break;
				default:
					break;
			}
			$this->response->redirect( $this->url->link( 'extension/export_import/tool/export_import', 'user_token='.$this->request->get['user_token']) );
		}

		$this->getForm();
	}


	public function settings() {
		$this->load->language('extension/export_import/tool/export_import');
		$this->document->setTitle($this->language->get('heading_title'));
		$this->load->model('extension/export_import/tool/export_import');
		if (($this->request->server['REQUEST_METHOD'] == 'POST') && ($this->validateSettingsForm())) {
			$this->load->model('setting/setting');
			$this->model_setting_setting->editSetting('export_import', $this->request->post);
			$this->session->data['success'] = $this->language->get('text_success_settings');
			$this->response->redirect($this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token']));
		}
		$this->getForm();
	}


	protected function getForm() {
		$data = array();

		$data['error_post_max_size'] = str_replace( '%1', ini_get('post_max_size'), $this->language->get('error_post_max_size') );
		$data['error_upload_max_filesize'] = str_replace( '%1', ini_get('upload_max_filesize'), $this->language->get('error_upload_max_filesize') );

		if (!empty($this->session->data['export_import_error']['errstr'])) {
			$this->error['warning'] = $this->session->data['export_import_error']['errstr'];
		} else if (isset($this->session->data['warning'])) {
			$this->error['warning'] = $this->session->data['warning'];
		}

 		if (isset($this->error['warning'])) {
			$data['error_warning'] = $this->error['warning'];
			if (!empty($this->session->data['export_import_nochange'])) {
				$data['error_warning'] .= "<br />\n".$this->language->get( 'text_nochange' );
			}
		} else {
			$data['error_warning'] = '';
		}

		unset($this->session->data['warning']);
		unset($this->session->data['export_import_error']);
		unset($this->session->data['export_import_nochange']);

		if (isset($this->session->data['success'])) {
			$data['success'] = $this->session->data['success'];
		
			unset($this->session->data['success']);
		} else {
			$data['success'] = '';
		}

		$data['breadcrumbs'] = array();
		$data['breadcrumbs'][] = array(
			'text' => $this->language->get('text_home'),
			'href' => $this->url->link('common/dashboard', 'user_token=' . $this->session->data['user_token'])
		);
		$data['breadcrumbs'][] = array(
			'text' => $this->language->get('heading_title'),
			'href' => $this->url->link('extension/export_import/tool/export_import', 'user_token=' . $this->session->data['user_token'])
		);

		$data['back'] = $this->url->link('common/dashboard', 'user_token=' . $this->session->data['user_token']);
		$data['button_back'] = $this->language->get( 'button_back' );
		$data['import'] = $this->url->link('extension/export_import/tool/export_import'.$this->method_separator.'upload', 'user_token=' . $this->session->data['user_token']);
		$data['export'] = $this->url->link('extension/export_import/tool/export_import'.$this->method_separator.'download', 'user_token=' . $this->session->data['user_token']);
		$data['settings'] = $this->url->link('extension/export_import/tool/export_import'.$this->method_separator.'settings', 'user_token=' . $this->session->data['user_token']);
		$data['post_max_size'] = $this->return_bytes( ini_get('post_max_size') );
		$data['upload_max_filesize'] = $this->return_bytes( ini_get('upload_max_filesize') );

		if (isset($this->request->post['export_type'])) {
			$data['export_type'] = $this->request->post['export_type'];
		} else {
			$data['export_type'] = 'p';
		}

		if (isset($this->request->post['range_type'])) {
			$data['range_type'] = $this->request->post['range_type'];
		} else {
			$data['range_type'] = 'id';
		}

		if (isset($this->request->post['min'])) {
			$data['min'] = $this->request->post['min'];
		} else {
			$data['min'] = '';
		}

		if (isset($this->request->post['max'])) {
			$data['max'] = $this->request->post['max'];
		} else {
			$data['max'] = '';
		}

		if (isset($this->request->post['incremental'])) {
			$data['incremental'] = $this->request->post['incremental'];
		} else {
			$data['incremental'] = '1';
		}

		if (isset($this->request->post['export_import_settings_use_option_id'])) {
			$data['settings_use_option_id'] = $this->request->post['export_import_settings_use_option_id'];
		} else if ($this->config->get( 'export_import_settings_use_option_id' )) {
			$data['settings_use_option_id'] = '1';
		} else {
			$data['settings_use_option_id'] = '0';
		}

		if (isset($this->request->post['export_import_settings_use_option_value_id'])) {
			$data['settings_use_option_value_id'] = $this->request->post['export_import_settings_use_option_value_id'];
		} else if ($this->config->get( 'export_import_settings_use_option_value_id' )) {
			$data['settings_use_option_value_id'] = '1';
		} else {
			$data['settings_use_option_value_id'] = '0';
		}

		if (isset($this->request->post['export_import_settings_use_attribute_group_id'])) {
			$data['settings_use_attribute_group_id'] = $this->request->post['export_import_settings_use_attribute_group_id'];
		} else if ($this->config->get( 'export_import_settings_use_attribute_group_id' )) {
			$data['settings_use_attribute_group_id'] = '1';
		} else {
			$data['settings_use_attribute_group_id'] = '0';
		}

		if (isset($this->request->post['export_import_settings_use_attribute_id'])) {
			$data['settings_use_attribute_id'] = $this->request->post['export_import_settings_use_attribute_id'];
		} else if ($this->config->get( 'export_import_settings_use_attribute_id' )) {
			$data['settings_use_attribute_id'] = '1';
		} else {
			$data['settings_use_attribute_id'] = '0';
		}

		if (isset($this->request->post['export_import_settings_use_filter_group_id'])) {
			$data['settings_use_filter_group_id'] = $this->request->post['export_import_settings_use_filter_group_id'];
		} else if ($this->config->get( 'export_import_settings_use_filter_group_id' )) {
			$data['settings_use_filter_group_id'] = '1';
		} else {
			$data['settings_use_filter_group_id'] = '0';
		}

		if (isset($this->request->post['export_import_settings_use_filter_id'])) {
			$data['settings_use_filter_id'] = $this->request->post['export_import_settings_use_filter_id'];
		} else if ($this->config->get( 'export_import_settings_use_filter_id' )) {
			$data['settings_use_filter_id'] = '1';
		} else {
			$data['settings_use_filter_id'] = '0';
		}

		$data['categories'] = array();
		$data['manufacturers'] = array();

		$min_product_id = $this->model_extension_export_import_tool_export_import->getMinProductId();
		$max_product_id = $this->model_extension_export_import_tool_export_import->getMaxProductId();
		$count_product = $this->model_extension_export_import_tool_export_import->getCountProduct();
		$min_category_id = $this->model_extension_export_import_tool_export_import->getMinCategoryId();
		$max_category_id = $this->model_extension_export_import_tool_export_import->getMaxCategoryId();
		$count_category = $this->model_extension_export_import_tool_export_import->getCountCategory();
		$min_customer_id = $this->model_extension_export_import_tool_export_import->getMinCustomerId();
		$max_customer_id = $this->model_extension_export_import_tool_export_import->getMaxCustomerId();
		$count_customer = $this->model_extension_export_import_tool_export_import->getCountCustomer();

		$data['text_welcome'] = str_replace('%1',$this->model_extension_export_import_tool_export_import->getVersion(),$this->language->get('text_welcome'));
		$data['text_used_category_ids'] = $this->language->get('text_used_category_ids');
		$data['text_used_category_ids'] = str_replace('%1',$min_category_id,$data['text_used_category_ids']);
		$data['text_used_category_ids'] = str_replace('%2',$max_category_id,$data['text_used_category_ids']);
		$data['text_used_product_ids'] = $this->language->get('text_used_product_ids');
		$data['text_used_product_ids'] = str_replace('%1',$min_product_id,$data['text_used_product_ids']);
		$data['text_used_product_ids'] = str_replace('%2',$max_product_id,$data['text_used_product_ids']);

		$data['version_export_import'] = $this->model_extension_export_import_tool_export_import->getVersion();
		$data['version_opencart'] = VERSION;
		$data['version_php'] = phpversion();

		$data['min_product_id'] = $min_product_id;
		$data['max_product_id'] = $max_product_id;
		$data['count_product'] = $count_product;
		$data['min_category_id'] = $min_category_id;
		$data['max_category_id'] = $max_category_id;
		$data['count_category'] = $count_category;
		$data['min_customer_id'] = $min_customer_id;
		$data['max_customer_id'] = $max_customer_id;
		$data['count_customer'] = $count_customer;

		$data['user_token'] = $this->session->data['user_token'];

		$data['method_separator'] = $this->method_separator;

		$this->document->addStyle('../extension/export_import/admin/view/stylesheet/export_import.css');

		$data['header'] = $this->load->controller('common/header');
		$data['column_left'] = $this->load->controller('common/column_left');
		$data['footer'] = $this->load->controller('common/footer');

		$view = $this->load->view( 'extension/export_import/tool/export_import', $data);
		$search = '<meta http-equiv="expires" content="0">';
		$add = '<script type="text/javascript">var export_import_alert = window.alert;</script>';
		$view = str_replace($search,$search."\n".$add,$view);
		$this->response->setOutput($view);
	}


	protected function validateDownloadForm() {
		if (!$this->user->hasPermission('access', 'extension/export_import/tool/export_import')) {
			$this->error['warning'] = $this->language->get('error_permission');
			return false;
		}

		if (!$this->config->get( 'export_import_settings_use_option_id' )) {
			$option_names = $this->model_extension_export_import_tool_export_import->getOptionNameCounts();
			foreach ($option_names as $option_name) {
				if ($option_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $option_name['name'], $this->language->get( 'error_option_name' ) );
					return false;
				}
			}
		}

		if (!$this->config->get( 'export_import_settings_use_option_value_id' )) {
			$option_value_names = $this->model_extension_export_import_tool_export_import->getOptionValueNameCounts();
			foreach ($option_value_names as $option_value_name) {
				if ($option_value_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $option_value_name['name'], $this->language->get( 'error_option_value_name' ) );
					return false;
				}
			}
		}

		if (!$this->config->get( 'export_import_settings_use_attribute_group_id' )) {
			$attribute_group_names = $this->model_extension_export_import_tool_export_import->getAttributeGroupNameCounts();
			foreach ($attribute_group_names as $attribute_group_name) {
				if ($attribute_group_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $attribute_group_name['name'], $this->language->get( 'error_attribute_group_name' ) );
					return false;
				}
			}
		}

		if (!$this->config->get( 'export_import_settings_use_attribute_id' )) {
			$attribute_names = $this->model_extension_export_import_tool_export_import->getAttributeNameCounts();
			foreach ($attribute_names as $attribute_name) {
				if ($attribute_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $attribute_name['name'], $this->language->get( 'error_attribute_name' ) );
					return false;
				}
			}
		}

		if (!$this->config->get( 'export_import_settings_use_filter_group_id' )) {
			$filter_group_names = $this->model_extension_export_import_tool_export_import->getFilterGroupNameCounts();
			foreach ($filter_group_names as $filter_group_name) {
				if ($filter_group_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $filter_group_name['name'], $this->language->get( 'error_filter_group_name' ) );
					return false;
				}
			}
		}

		if (!$this->config->get( 'export_import_settings_use_filter_id' )) {
			$filter_names = $this->model_extension_export_import_tool_export_import->getFilterNameCounts();
			foreach ($filter_names as $filter_name) {
				if ($filter_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $filter_name['name'], $this->language->get( 'error_filter_name' ) );
					return false;
				}
			}
		}

		return true;
	}


	protected function validateUploadForm() {
		if (!$this->user->hasPermission('modify', 'extension/export_import/tool/export_import')) {
			$this->error['warning'] = $this->language->get('error_permission');
		} else if (!isset( $this->request->post['incremental'] )) {
			$this->error['warning'] = $this->language->get( 'error_incremental' );
		} else if ($this->request->post['incremental'] != '0') {
			if ($this->request->post['incremental'] != '1') {
				$this->error['warning'] = $this->language->get( 'error_incremental' );
			}
		}

		if (!isset($this->request->files['upload']['name'])) {
			if (isset($this->error['warning'])) {
				$this->error['warning'] .= "<br /\n" . $this->language->get( 'error_upload_name' );
			} else {
				$this->error['warning'] = $this->language->get( 'error_upload_name' );
			}
		} else {
			$ext = strtolower(pathinfo($this->request->files['upload']['name'], PATHINFO_EXTENSION));
			if (($ext != 'xls') && ($ext != 'xlsx') && ($ext != 'ods')) {
				if (isset($this->error['warning'])) {
					$this->error['warning'] .= "<br /\n" . $this->language->get( 'error_upload_ext' );
				} else {
					$this->error['warning'] = $this->language->get( 'error_upload_ext' );
				}
			}
		}

		if (!$this->error) {
			return true;
		} else {
			return false;
		}
	}


	protected function validateSettingsForm() {
		if (!$this->user->hasPermission('access', 'extension/export_import/tool/export_import')) {
			$this->error['warning'] = $this->language->get('error_permission');
			return false;
		}

		if (empty($this->request->post['export_import_settings_use_option_id'])) {
			$option_names = $this->model_extension_export_import_tool_export_import->getOptionNameCounts();
			foreach ($option_names as $option_name) {
				if ($option_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $option_name['name'], $this->language->get( 'error_option_name' ) );
					return false;
				}
			}
		}

		if (empty($this->request->post['export_import_settings_use_option_value_id'])) {
			$option_value_names = $this->model_extension_export_import_tool_export_import->getOptionValueNameCounts();
			foreach ($option_value_names as $option_value_name) {
				if ($option_value_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $option_value_name['name'], $this->language->get( 'error_option_value_name' ) );
					return false;
				}
			}
		}

		if (empty($this->request->post['export_import_settings_use_attribute_group_id'])) {
			$attribute_group_names = $this->model_extension_export_import_tool_export_import->getAttributeGroupNameCounts();
			foreach ($attribute_group_names as $attribute_group_name) {
				if ($attribute_group_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $attribute_group_name['name'], $this->language->get( 'error_attribute_group_name' ) );
					return false;
				}
			}
		}

		if (empty($this->request->post['export_import_settings_use_attribute_id'])) {
			$attribute_names = $this->model_extension_export_import_tool_export_import->getAttributeNameCounts();
			foreach ($attribute_names as $attribute_name) {
				if ($attribute_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $attribute_name['name'], $this->language->get( 'error_attribute_name' ) );
					return false;
				}
			}
		}

		if (empty($this->request->post['export_import_settings_use_filter_group_id'])) {
			$filter_group_names = $this->model_extension_export_import_tool_export_import->getFilterGroupNameCounts();
			foreach ($filter_group_names as $filter_group_name) {
				if ($filter_group_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $filter_group_name['name'], $this->language->get( 'error_filter_group_name' ) );
					return false;
				}
			}
		}

		if (empty($this->request->post['export_import_settings_use_filter_id'])) {
			$filter_names = $this->model_extension_export_import_tool_export_import->getFilterNameCounts();
			foreach ($filter_names as $filter_name) {
				if ($filter_name['count'] > 1) {
					$this->error['warning'] = str_replace( '%1', $filter_name['name'], $this->language->get( 'error_filter_name' ) );
					return false;
				}
			}
		}

		return true;
	}


	public function getNotifications() {
		sleep(1); // give the data some "feel" that its not in our system
		$this->load->model('extension/export_import/tool/export_import');
		$this->load->language( 'extension/export_import/tool/export_import' );
		$response = $this->model_extension_export_import_tool_export_import->getNotifications();
		$json = array();
		if ($response===false) {
			$json['message'] = '';
			$json['error'] = $this->language->get( 'error_notifications' );
		} else {
			$json['message'] = $response;
			$json['error'] = '';
		}
		$this->response->setOutput(json_encode($json));
	}


	public function getCountProduct() {
		$this->load->model('extension/export_import/tool/export_import');
		$count = $this->model_extension_export_import_tool_export_import->getCountProduct();
		$json = array( 'count'=>$count );
		$this->response->addHeader('Content-Type: application/json');
		$this->response->setOutput(json_encode($json));
	}
}
?>
