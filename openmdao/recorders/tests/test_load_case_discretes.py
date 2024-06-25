""" Unit tests that exercise Case via the Problem.load_case method. """

import unittest

import numpy as np

import openmdao.api as om
from openmdao.recorders.tests.recorder_test_utils import assert_model_matches_case
from openmdao.utils.testing_utils import use_tempdirs


class SellarMDAWithUnits(om.Group):
    """
    Group containing the Sellar MDA.
    """

    class SellarDis1Units(om.ExplicitComponent):
        """
        Component containing Discipline 1 -- no derivatives version.
        """

        def setup(self):

            # Global Design Variable
            self.add_input('z', val=np.zeros(2), units='degC')

            # Local Design Variable
            self.add_input('x', val=0., units='degC')

            # Coupling parameter
            self.add_input('y2', val=1.0, units='degC')

            # Coupling output
            self.add_output('y1', val=1.0, units='degC')

            self.add_discrete_output('disc_out', val=['a'])

        def setup_partials(self):
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')

        def compute(self, inputs, outputs, dins, douts):
            """
            Evaluates the equation
            y1 = z1**2 + z2 + x1 - 0.2*y2
            """
            z1 = inputs['z'][0]
            z2 = inputs['z'][1]
            x1 = inputs['x']
            y2 = inputs['y2']

            outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2


    class SellarDis2Units(om.ExplicitComponent):
        """
        Component containing Discipline 2 -- no derivatives version.
        """

        def setup(self):
            # Global Design Variable
            self.add_input('z', val=np.zeros(2), units='degC')

            # Coupling parameter
            self.add_input('y1', val=1.0, units='degC')

            # Coupling output
            self.add_output('y2', val=1.0, units='degC')

        def setup_partials(self):
            # Finite difference all partials.
            self.declare_partials('*', '*', method='fd')

        def compute(self, inputs, outputs):
            """
            Evaluates the equation
            y2 = y1**(.5) + z1 + z2
            """

            z1 = inputs['z'][0]
            z2 = inputs['z'][1]
            y1 = inputs['y1']

            # Note: this may cause some issues. However, y1 is constrained to be
            # above 3.16, so lets just let it converge, and the optimizer will
            # throw it out
            if y1.real < 0.0:
                y1 *= -1

            outputs['y2'] = y1**.5 + z1 + z2

    def setup(self):

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', self.SellarDis1Units(), promotes_inputs=['x', 'z', 'y2'],
                            promotes_outputs=['y1'])
        cycle.add_subsystem('d2', self.SellarDis2Units(), promotes_inputs=['z', 'y1'],
                            promotes_outputs=['y2'])

        cycle.set_input_defaults('x', 1.0, units='degC')
        cycle.set_input_defaults('z', np.array([5.0, 2.0]), units='degC')

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z={'val': np.array([0.0, 0.0]), 'units': 'degC'},
                                                  x={'val': 0.0, 'units': 'degC'},
                                                  y1={'units': 'degC'},
                                                  y2={'units': 'degC'}),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', y1={'units': 'degC'},
                                                   con1={'units': 'degC'}),
                           promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', y2={'units': 'degC'},
                                                   con2={'units': 'degC'}),
                           promotes=['con2', 'y2'])


@use_tempdirs
class TestLoadCase(unittest.TestCase):

    def test_load_case_with_discretes(self):
        # build the model
        prob = om.Problem(model=SellarMDAWithUnits())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        # Attach a recorder to the problem
        recorder = om.SqliteRecorder('cases.sql')
        prob.add_recorder(recorder)
        prob.recording_options['record_inputs'] = True

        # run driver and record case
        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.record("after_run_driver")

        # get recorded case and load it back into the model
        cr = om.CaseReader("cases.sql")
        case = cr.get_case('after_run_driver')
        prob.load_case(case)

        assert_model_matches_case(case, model)


if __name__ == "__main__":
    unittest.main()
