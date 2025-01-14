import AppBar from "@mui/material/AppBar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import GlobalStyles from "@mui/material/GlobalStyles";
import { useNavigate } from "react-router-dom";


const Header = () => {
  const navigate = useNavigate();
  const routeChange = () => {
    window.location.href = 'http://localhost:8501/';
  }
  return (
    <>
      <GlobalStyles
        styles={{ ul: { margin: 0, padding: 0, listStyle: "none" } }}
      />
      <CssBaseline />
      <AppBar
        position="static"
        color="default"
        elevation={0}
        sx={{ borderBottom: (theme) => `1px solid ${theme.palette.divider}` }}
      >
        <Toolbar sx={{ flexWrap: "wrap" }}>
          <Typography
            variant="h5"
            ml={2}
            color="inherit"
            noWrap
            sx={{ flexGrow: 1 }}
          >
            <b>WarpLearn</b>

          </Typography>
          <Button
            color="secondary"
            variant="outlined"
            sx={{ my: 1, mx: 1.5 }}
            onClick={routeChange}
          >
            Get Started
          </Button>
        </Toolbar>
      </AppBar>
    </>
  );
};

export default Header;
