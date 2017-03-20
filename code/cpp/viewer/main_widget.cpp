//
// Created by yanhang on 3/19/17.
//

#include "main_widget.h"

using namespace std;

namespace IMUProject{

	MainWidget::MainWidget(const std::string &path, QWidget *parent) {
		navigation_.reset(new Navigation());
	}

	void MainWidget::initializeGL() {
		initializeOpenGLFunctions();

		cout << "Initializing the canvas..." << endl;
		canvas_.reset(new Canvas(100, 100));

		glClearColor(0.f, 0.f, 0.f, 1.f);
		glEnable(GL_DEPTH);
		//glClearDepth(0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		AllocateRecourse();

		glViewport(0, 0, width(), height());
	}

	void MainWidget::resizeGL(int w, int h) {

	}

	void MainWidget::paintGL() {

	}

	void MainWidget::timerEvent(QTimerEvent *event) {

	}

	void MainWidget::Start() {

	}

	void MainWidget::Stop(){

	}

	void MainWidget::AllocateRecourse() {

	}

	void MainWidget::FreeResource() {

	}

}//namespace IMUProject