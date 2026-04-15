<?php
namespace Opencart\Admin\Controller\Extension\Recommendation\Module;

class Recommendation extends \Opencart\System\Engine\Controller {

    public function index(): void {
        $this->load->language('extension/recommendation/module/recommendation');
        $this->document->setTitle($this->language->get('heading_title'));

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
            'href' => $this->url->link('extension/recommendation/module/recommendation', 'user_token=' . $this->session->data['user_token'])
        ];

        $data['save'] = $this->url->link('extension/recommendation/module/recommendation.save', 'user_token=' . $this->session->data['user_token']);
        $data['back'] = $this->url->link('marketplace/extension', 'user_token=' . $this->session->data['user_token'] . '&type=module');

        $data['module_recommendation_status'] = isset($this->request->post['module_recommendation_status']) ? $this->request->post['module_recommendation_status'] : $this->config->get('module_recommendation_status');

        $data['header'] = $this->load->controller('common/header');
        $data['column_left'] = $this->load->controller('common/column_left');
        $data['footer'] = $this->load->controller('common/footer');

        $this->response->setOutput($this->load->view('extension/recommendation/module/recommendation', $data));
    }

    public function save(): void {
        $this->load->language('extension/recommendation/module/recommendation');

        $json = [];

        if (!$this->user->hasPermission('modify', 'extension/recommendation/module/recommendation')) {
            $json['error']['warning'] = $this->language->get('error_permission');
        }

        if (!$json) {
            $this->load->model('setting/setting');
            $this->model_setting_setting->editSetting('module_recommendation', $this->request->post);

            $json['success'] = $this->language->get('text_success');
        }

        $this->response->addHeader('Content-Type: application/json');
        $this->response->setOutput(json_encode($json));
    }

    public function install(): void {
    }

    public function uninstall(): void {
        $this->db->query("DELETE FROM \`" . DB_PREFIX . "setting\` WHERE code = 'module_recommendation'");
    }
}