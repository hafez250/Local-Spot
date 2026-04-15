<?php
namespace Opencart\Admin\Controller\Extension\Opencart\Module;

class Recommendation extends \Opencart\System\Engine\Controller {

    public function index(): void {
        // Load language
        $this->load->language('extension/opencart/module/recommendation');

        // Page title
        $this->document->setTitle($this->language->get('heading_title'));

        // Breadcrumbs
        $data['breadcrumbs'] = [];
        $data['breadcrumbs'][] = [
            'text' => $this->language->get('text_home'),
            'href' => $this->url->link('common/dashboard', 'user_token=' . $this->session->data['user_token'])
        ];
        $data['breadcrumbs'][] = [
            'text' => $this->language->get('text_extension'),
            'href' => $this->url->link('marketplace/extension', 'user_token=' . $this->session->data['user_token'] . '&type=module')
        ];
        $data['breadcrumbs'][] = [
            'text' => $this->language->get('heading_title'),
            'href' => $this->url->link('extension/opencart/module/recommendation', 'user_token=' . $this->session->data['user_token'])
        ];

        // Save & back URLs
        $data['save'] = $this->url->link(
            'extension/opencart/module/recommendation/save', 
            'user_token=' . $this->session->data['user_token'] . (isset($this->request->get['module_id']) ? '&module_id=' . (int)$this->request->get['module_id'] : '')
        );
        $data['back'] = $this->url->link('marketplace/extension', 'user_token=' . $this->session->data['user_token'] . '&type=module');

        // Load module model
        $this->load->model('setting/module');

        // Load existing module data if editing
        if (isset($this->request->get['module_id'])) {
            $module_info = $this->model_setting_module->getModule($this->request->get['module_id']);
        } else {
            $module_info = [];
        }

        // Set default values
        $data['module_recommendation_name'] = $this->request->post['module_recommendation_name'] ?? $module_info['name'] ?? 'Recommendation';
        $data['module_recommendation_category_name'] = $this->request->post['module_recommendation_category_name'] ?? $module_info['category_name'] ?? '';
        $data['module_recommendation_number_of_products'] = $this->request->post['module_recommendation_number_of_products'] ?? $module_info['number_of_products'] ?? 5;
        $data['module_recommendation_sort_by'] = $this->request->post['module_recommendation_sort_by'] ?? $module_info['sort_by'] ?? 'date_added';
        $data['module_recommendation_status'] = $this->request->post['module_recommendation_status'] ?? $module_info['status'] ?? 0;

        // Common sections
        $data['header'] = $this->load->controller('common/header');
        $data['column_left'] = $this->load->controller('common/column_left');
        $data['footer'] = $this->load->controller('common/footer');

        // Render template
        $this->response->setOutput($this->load->view('extension/opencart/module/recommendation', $data));
    }

    public function save(): void {
        $this->load->language('extension/opencart/module/recommendation');

        $json = [];

        // Permission check
        if (!$this->user->hasPermission('modify', 'extension/opencart/module/recommendation')) {
            $json['error'] = $this->language->get('error_permission');
        }

        if (!$json) {
            // Load module model
            $this->load->model('setting/module');

            $data = [
                'name' => $this->request->post['module_recommendation_name'] ?? 'Recommendation',
                'category_name' => $this->request->post['module_recommendation_category_name'] ?? '',
                'number_of_products' => $this->request->post['module_recommendation_number_of_products'] ?? 5,
                'sort_by' => $this->request->post['module_recommendation_sort_by'] ?? 'date_added',
                'status' => $this->request->post['module_recommendation_status'] ?? 0
            ];

            // If module_id exists, edit module, else add new
            if (isset($this->request->get['module_id'])) {
                $this->model_setting_module->editModule((int)$this->request->get['module_id'], $data);
            } else {
                $this->model_setting_module->addModule('recommendation', $data);
            }

            $json['success'] = $this->language->get('text_success');
        }

        // Return JSON
        $this->response->addHeader('Content-Type: application/json');
        $this->response->setOutput(json_encode($json));
    }
}
