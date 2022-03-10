/*
    Copyright 2016-2022 melonDS team

    This file is part of melonDS.

    melonDS is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    melonDS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with melonDS. If not, see http://www.gnu.org/licenses/.
*/

#ifndef TITLEMANAGERDIALOG_H
#define TITLEMANAGERDIALOG_H

#include <QDialog>
#include <QMessageBox>
#include <QListWidget>
#include <QButtonGroup>

#include <QUrl>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QNetworkAccessManager>

namespace Ui
{
    class TitleManagerDialog;
    class TitleImportDialog;
}
class TitleManagerDialog;
class TitleImportDialog;

class TitleManagerDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TitleManagerDialog(QWidget* parent);
    ~TitleManagerDialog();

    static FILE* curNAND;
    static bool openNAND();
    static void closeNAND();

    static TitleManagerDialog* currentDlg;
    static TitleManagerDialog* openDlg(QWidget* parent)
    {
        if (currentDlg)
        {
            currentDlg->activateWindow();
            return currentDlg;
        }

        if (!openNAND())
        {
            QMessageBox::critical(parent,
                                  "DSi title manager - melonDS",
                                  "Failed to mount the DSi NAND. Check that your NAND dump is accessible and valid.");
            return nullptr;
        }

        currentDlg = new TitleManagerDialog(parent);
        currentDlg->open();
        return currentDlg;
    }
    static void closeDlg()
    {
        currentDlg = nullptr;
        closeNAND();
    }

private slots:
    void done(int r);

    void on_btnImportTitle_clicked();
    void onImportTitleFinished(int res);
    void on_btnDeleteTitle_clicked();
    void on_lstTitleList_currentItemChanged(QListWidgetItem* cur, QListWidgetItem* prev);
    void onImportTitleData();
    void onExportTitleData();

private:
    Ui::TitleManagerDialog* ui;

    QString importAppPath;
    u8 importTmdData[0x208];
    bool importReadOnly;

    QAction* actImportTitleData[3];
    QAction* actExportTitleData[3];

    void createTitleItem(u32 category, u32 titleid);
};

class TitleImportDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TitleImportDialog(QWidget* parent, QString& apppath, u8* tmd, bool& readonly);
    ~TitleImportDialog();

private slots:
    void accept() override;
    void tmdDownloaded();

    void on_btnAppBrowse_clicked();
    void on_btnTmdBrowse_clicked();
    void onChangeTmdSource(int id);

private:
    Ui::TitleImportDialog* ui;

    QButtonGroup* grpTmdSource;

    QNetworkAccessManager* network;
    QNetworkReply* netreply;

    QString& appPath;
    u8* tmdData;
    bool& readOnly;

    u32 titleid[2];
};

#endif // TITLEMANAGERDIALOG_H
